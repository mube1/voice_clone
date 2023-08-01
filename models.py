import torch


# For the generator
class AudioAutoencoder(torch.nn.Module):
    def __init__(self, input_size=84224, hidden_size=1028, output_size=84224):
        super(AudioAutoencoder, self).__init__()
   
        self.encoder = torch.nn.Sequential(            
            nn.Linear(input_size, hidden_size),
            
            torch.nn.Unflatten(-1,(1,1,hidden_size)),
            torch.nn.BatchNorm1d(1),
            torch.nn.ReLU(),
            torch.nn.Flatten(1),
            
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Unflatten(-1,(1,1,hidden_size)),
     
            torch.nn.Linear(hidden_size, 512)
        )
    
   ##############################################################################   
    
        self.decoder = nn.Sequential(
            torch.nn.Flatten(start_dim=0),
            torch.nn.Linear(512, hidden_size),
            torch.nn.Unflatten(-1,(1,1,hidden_size)),            
            torch.nn.BatchNorm1d(1),
            torch.nn.ReLU(),
            # nn.Flatten(1),
            torch.nn.Linear(hidden_size, hidden_size),
            # nn.Unflatten(-1,(1,1,hidden_size)),
            torch.nn.BatchNorm1d(1),
            torch.nn.ReLU(), 
            torch.nn.Linear(hidden_size, output_size),
            torch.nn.Flatten(start_dim=0),
        )

    def forward(self, x):
        z = self.encoder(x)        
        reconstructed_x = self.decoder(z)
        return reconstructed_x


# For the discriminator


class PatchGANDiscriminator(torch.nn.Module):
    def __init__(self):
        super(PatchGANDiscriminator, self).__init__()
        self.conv = torch.nn.Conv2d(1, 64, kernel_size=4, stride=2,bias=False)
        self.conv1 = torch.nn.Conv2d(64, 128, kernel_size=4, stride=2,bias=False)        
        self.conv2 = torch.nn.Conv2d(128, 256, kernel_size=4, stride=2, bias=False)
        self.conv3 = torch.nn.Conv2d(256, 512, kernel_size=4, stride=2,bias=False)
#         self.pool= torch.nn.MaxPool2d(3, stride=2)
        
        self.conv4=torch.nn.Conv2d(in_channels=512,out_channels=1,kernel_size=4,stride=1,  padding=1  )

    def forward(self, x):
        x = self.conv(x)
        x = torch.relu(x)        
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = torch.relu(x)
        x = self.conv4(x)

        return x
