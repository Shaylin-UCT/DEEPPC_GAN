'''Assumptions: 
1. Always enter the name of the subfolder in ImagesforResearch'''

class DataPrep():

    def __init__(self, element, img_size):
        #Element = Elbow, Neck, etc. 
        self.element = element
        self.img_size = img_size

    def getData(self):
        #Imports
        import pathlib
        from torchvision import datasets
        import torch
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms
        import os

        data_path = pathlib.Path("./ImagesforResearch")
        #data_path = pathlib.Path("C:/Users/shayl/OneDrive/Documents/Honors/Research/CodeBase/TrainingData")
        image_path = data_path / self.element
        #print(image_path)
        if image_path.is_dir(): #Check if directory exists
            print("Currently working in:", os.getcwd())
            print(f"{image_path} directory exists.")
            pass
        else:
            print("Currently working in:", os.getcwd())
            print(f"{image_path} directory DOES NOT exist.")
        data_transform = transforms.Compose(
            [
            # Resize the images to 64x64
            transforms.Resize((self.img_size, self.img_size)),
            # Turn the image into a torch.Tensor
            transforms.ToTensor(), # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0
            #Normalize the tensor
            transforms.Normalize([0.5], [0.5])
            ]
            )
        #Get all images
        image_path_list = list(image_path.glob("*.jpg")) 

        #Create new dataset:
        newData =  datasets.ImageFolder(root=data_path, transform=data_transform,target_transform=None)
        #print(f"Dataset consists of:\n{newData}")
        #Return data to be used in a dataloader
        return newData


def main():
    from torch.utils.data import DataLoader
    x = DataPrep("Elbow", 28)
    hold = x.getData()
    dataloader = DataLoader(dataset=x.getData(), 
                                batch_size=64, # how many samples per batch?
                                num_workers=1, # how many subprocesses to use for data loading? (higher = more)
                                shuffle=True) # shuffle the data?
    print(dataloader)
    print("len(dataloader):",len(dataloader))
    print("batch size:",dataloader.batch_size)
    print("sampler:", dataloader.sampler)

    import torch
    cuda = True if torch.cuda.is_available() else False

    for i, (imgs, _) in enumerate(dataloader):
        # Configure input
        #Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        #real_imgs = Variable(imgs.type(Tensor))
        #hold = imgs.type(Tensor)
        #print("Type of \"hold\"", type(hold))
        #print("Hold.size:",hold.size())
        print("Batch:", i)
        print("BatchSize:",imgs.size())
        print("BatchLabel:", _, _.size()) #The label for the image. It is assigned a value in our case
        #break #Just done to print the first batch only

if __name__=="__main__":
    main()