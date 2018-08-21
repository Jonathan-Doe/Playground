class Generator(nn.Module):
    def __init__(self):
        
        super(Generator, self).__init__()
        self.init_size = 250
        self.init_channels = 128
        self.l1 = nn.Sequential(nn.Linear(latent_dim , self.init_channels * self.init_size))
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm1d(self.init_channels),
            #nn.Upsample(scale_factor=2),
            nn.Conv1d(128, 128, 5, stride=1, padding=1),
            nn.BatchNorm1d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Upsample(scale_factor=2),
            nn.Conv1d(128, 256, 5, stride=1, padding=1),
            nn.BatchNorm1d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(256, channels, 3, stride=1, padding=3),
            nn.Tanh()
        )
        

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], self.init_channels,self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            if bn:
                block = nn.Sequential( nn.Conv1d(in_filters, out_filters, 3, 2, 1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Dropout(0.5),
                        nn.BatchNorm1d(out_filters, 0.8))
            
            else:
                block = nn.Sequential( nn.Conv1d(in_filters, out_filters, 3, 2, 1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Dropout(0.5))
                
            return block

        self.model = nn.Sequential(
            discriminator_block(channels, 16, bn=False),
            discriminator_block(16, 32),
            discriminator_block(32, 64),
            discriminator_block(64, 128),)
        

        # The height and width of downsampled image
        ds_size = 2048
        
        self.adv_layer = nn.Sequential( nn.Linear(ds_size, 1),
                                        nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity