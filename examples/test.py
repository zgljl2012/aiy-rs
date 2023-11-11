import torch

if __name__ == '__main__':
    a = list([1,2] + [3, 4] + [5, 6])
    b = torch.tensor(a)
    print(b.reshape(1, -1).shape)
