import torch
import unittest
from Utils import rank_acc, create_unseen_embds
from MoEViT import cos_face_loss
import torch.nn as nn
import math
from Attention import MQGAttention, precompute_theta_pos_frequencies, apply_rotary_embeddings, repeat_kv
from MoEViT import MQGAttention, MoEViTConfig
from torch.nn import functional as F
import torchtune

class MockModel(nn.Module):
    def __init__(self):
        super(MockModel, self).__init__()
        self.l1 = nn.Conv2d(3,3,1)
    
    def forward(self, images, labels):   
        self.l1(images)
        # Create mock logits, embeddings and loss
        logits = torch.randn(len(labels), 10)  # Assume 10 classes
        embeddings = torch.randn(len(labels), 128)  # Assume embedding size of 128
        loss = torch.tensor(0.1)  # Constant loss
        return logits, embeddings, loss


def angle_between_vectors(v1, v2):
    """
    Calculate the angle in degrees between two vectors using PyTorch.

    Args:
    v1 (torch.Tensor): The first vector.
    v2 (torch.Tensor): The second vector.

    Returns:
    float: The angle between the vectors in degrees.
    """
    # Calculate the dot product of the vectors
    dot_product = torch.dot(v1, v2)
    
    # Calculate the magnitudes of the vectors
    magnitude_v1 = torch.norm(v1)
    magnitude_v2 = torch.norm(v2)
    
    # Calculate the cosine of the angle using the dot product formula
    cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
    
    # Ensure the value does not fall outside the domain for arccos due to floating point errors
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    
    # Calculate the angle in radians 
    angle = torch.arccos(cos_theta) 
    
    return angle.item()

class TestRankAccuracy(unittest.TestCase):
    
    def test_perfect_accuracy(self):
        # Create embeddings and labels
        embeddings = torch.tensor([
            [1.0, 0.0],
            [0.1, 1.0],
            [0.2, 1.0],
            [1.1, 0.1],
            [1.0, 1.0],
            [1.1, 1.1]
        ])
        labels = torch.tensor([0, 1, 1, 0, 2, 2])

        expected_rank_1_acc = 1.0  #
        expected_rank_5_acc = 1.0  # 

        # Call the function
        rank_1_acc, rank_5_acc = rank_acc(embeddings, labels)

        # Assert the conditions
        assert rank_1_acc == expected_rank_1_acc, "Rank 1 accuracy does not match expected"
        assert rank_5_acc == expected_rank_5_acc, "Rank 5 accuracy does not match expected"


    def test_null_accuracy(self):
        # Create embeddings and labels
        embeddings = torch.tensor([
            [1.0, 0.0],
            [0.1, 1.0],
            [0.2, 1.0],
            [1.1, 0.1],
            [1.0, 1.0],
            [1.1, 1.1],
        ])
        labels = torch.tensor([0, 0, 1, 2, 1, 2])

        expected_rank_1_acc = 0.0  #
        expected_rank_5_acc = 1.0  # 

        # Call the function
        rank_1_acc, rank_5_acc = rank_acc(embeddings, labels)

        # Assert the conditions
        assert rank_1_acc == expected_rank_1_acc, "Rank 1 accuracy does not match expected"
        assert rank_5_acc == expected_rank_5_acc, "Rank 5 accuracy does not match expected"

    

class TestCreateUnseenEmbds(unittest.TestCase):

    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_create_unseen_embds(self):
        model = MockModel()

        # Creating a DataLoader with batch size of 2
        images = torch.randn(4, 3, 64, 64)  # 4 images of size 64x64 with 3 color channels
        labels = torch.tensor([0, 1, 0, 1])  # Corresponding labels
        dataset = torch.utils.data.TensorDataset(images, labels)
        loader = torch.utils.data.DataLoader(dataset, batch_size=2)

        # Testing the function
        embds, labels = create_unseen_embds(model, loader, self.device)

        # Assert the conditions
        assert embds.shape == (4,128), "embeddings shape does not match expected"
        assert labels.shape == (4,), "labels shape does not match expected"

class TestCosFaceLoss(unittest.TestCase):
    
    def setUp(self):
        # Assuming CosFaceLoss is imported correctly
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_one_vector_two_classes(self):
        X = torch.tensor([[1.0, 2.0]], device=self.device)
        W = torch.tensor([[1.0, 2.0], [2.0, 3.0]], device=self.device)
        y = torch.tensor([0], device=self.device)
        s = 10
        m = 0.1
        loss = cos_face_loss(X, W, s, m, y, self.device)

        angle_0_0 = angle_between_vectors(X[0,:],W[0,:])
        angle_0_1 = angle_between_vectors(X[0,:],W[1,:])
        exp_num = math.exp(10*(math.cos(angle_0_0)-m))
        exp_neg_w = math.exp(10*(math.cos(angle_0_1)))
        expected_loss_value = -math.log(exp_num / (exp_neg_w+exp_num))
        self.assertAlmostEqual(loss.item(), expected_loss_value, places=4)

    def test_two_vectors_two_classes(self):
        X = torch.tensor([[1.0, 2.0], [2.5, 3.5]], device=self.device)
        W = torch.tensor([[1.0, 2.0], [2.0, 3.0]], device=self.device)
        y = torch.tensor([0,1], device=self.device)
        s = 10
        m = 0.1
        loss = cos_face_loss(X, W, s, m, y, self.device)

        angle_0_0 = angle_between_vectors(X[0,:],W[0,:])
        angle_0_1 = angle_between_vectors(X[0,:],W[1,:])
        angle_1_0 = angle_between_vectors(X[1,:],W[0,:])
        angle_1_1 = angle_between_vectors(X[1,:],W[1,:])      
        exp_num_0 = math.exp(10*(math.cos(angle_0_0)-m))
        exp_num_1 = math.exp(10*(math.cos(angle_1_1)-m))
        exp_neg_w_0 = math.exp(10*(math.cos(angle_0_1)))
        exp_neg_w_1 = math.exp(10*(math.cos(angle_1_0)))
        expected_loss_value = -0.5 * (math.log(exp_num_0 / (exp_neg_w_0+exp_num_0)) + math.log(exp_num_1 / (exp_neg_w_1+exp_num_1)))
        self.assertAlmostEqual(loss.item(), expected_loss_value, places=4)


class TestAttention(unittest.TestCase):
    
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_rotary_embeddings(self):

        batch, sequence, head_dim, num_heads = 1, 20, 4, 1

        tensor = torch.randn((batch, sequence, num_heads, head_dim)).to(self.device)

        freqs_complex = precompute_theta_pos_frequencies(head_dim=head_dim, seq_len=sequence, device=self.device)

        rotated_tensor = apply_rotary_embeddings(tensor, freqs_complex, device=self.device)
        
        rotation_embd_pytorch = torchtune.modules.RotaryPositionalEmbeddings(head_dim,sequence).to(self.device)

        rotated_tensor_pytorch = rotation_embd_pytorch(tensor)

        self.assertAlmostEqual(rotated_tensor.all(), rotated_tensor_pytorch.all())

    def test_attention(self):

        img_dim, patch_size = 8, 2

        batch, sequence, embd_dim, num_q_heads, num_k_heads = 1, int((img_dim/patch_size)**2), 8, 4, 2

        head_dim = embd_dim // num_q_heads

        config = MoEViTConfig(n_heads=num_q_heads, n_kv_heads=num_k_heads, n_embd=embd_dim)

        att = MQGAttention(config).to(self.device)

        x = torch.randn((batch, sequence, embd_dim)).to(self.device)

        freqs_complex = precompute_theta_pos_frequencies(head_dim=head_dim, seq_len=sequence, device=self.device)

        results_att = att(x, freqs_complex)

        assert results_att.shape == (batch, sequence, embd_dim)

    
    def test_repeat_kv(self):

        #(B, Seq_Len_KV, H_KV, Head_Dim)
        batch, seq, embd_dim, num_q_heads, num_kv_heads = 1, 2, 12, 6, 3

        n_rep = num_q_heads // num_kv_heads

        x = torch.randn((batch, seq, num_kv_heads, embd_dim//num_q_heads))

        x_new = repeat_kv(x, n_rep)

        for i in range(n_rep):

            assert x[0,0,0,:].all() == x_new[0,0,i,:].all()





if __name__ == '__main__':
    unittest.main()