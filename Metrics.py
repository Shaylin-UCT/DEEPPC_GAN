'''
-----------------------------------------------------------------------
Implementations of different GAN metrics using pytorch unless specified
-----------------------------------------------------------------------
'''
#Libraries
from pytorch_fid import fid_score, inception #FID
'''
                            --- FID ---

For FID, run command: python -m pytorch_fid <imageset1> <imageset2>
Could just include this in the shell script

Can include in the code using:
paths = [...] -> list of file paths
    batch_size = 50 -> number of samples in each path should be a multiple of this. If not,
        some would just be excluded
    device = "cpu" -> include a cuda check here
    dims = 2048 -> dims in image
    fid = fid_score.calculate_fid_given_paths(paths=paths, batch_size=batch_size, 
                                              device=device, dims = dims)
    print("FID", fid)
'''

def calculateFID(dataset1, dataset2):
    paths = [dataset1, dataset2]
    batch_size = 50
    device = "cpu"
    dims = 2048
    num_workers = 8 
    fid = fid_score.calculate_fid_given_paths(paths=paths, batch_size=batch_size, device=device, dims = dims)
    return fid

def main():
    fid = calculateFID("DeleteThis\\test1", "DeleteThis\\test2")
    print("FID", fid)

if __name__=="__main__":
    main()

