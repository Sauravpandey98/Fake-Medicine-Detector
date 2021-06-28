import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
import argparse



def load_model(model_path):
    """Function to covert trained model to Interpreter

    Parameter
            model_path: path of the trained model
    
    return
            Interpreter: Interpreter object """
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def model_output(img_path,interpreter):
    """A function to encode the image into a 256 embedding vetor using trained model
    
    Paramters
           img_path: path of imput image 
           interpreter: Interpreter object 

    return
          embedding: embedding of input image encoded by model"""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    img1 = Image.open(img_path).resize((width, height))
    img1=np.array(img1,dtype="float32")

    input1_data = np.expand_dims(img1, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input1_data)
    interpreter.invoke()
    embedding= interpreter.get_tensor(output_details[0]['index'])
    return embedding
        
    

def verify(img1_path,img2_path,model_path):
    """A function to calculate distance between embeddings of two images
    
    Paramters
            img1_path=path of first image
            img2_path=path of second image
            model_path=path of trained model
            
    return
        None"""
    interpreter=load_model(model_path)
    embedding1 = model_output(img1_path, interpreter)
    embedding2 = model_output(img2_path, interpreter)
    dist = np.linalg.norm(embedding1-embedding2)
    print("Calculated Euclidean distance between images of packages of both medicines is {}".format(dist))
    if(dist<=0.2):
        print("MUT is Genuine")
    else:
        print("MUT is fake")

def main():
    my_parser = argparse.ArgumentParser(description='Detect fake medicinde by comparing images of packages of original medicine and medicine under test(MUT).')

    my_parser.add_argument("model_path",metavar="Model_path",type=str,help="Path of trained model")
    my_parser.add_argument("original_img_path",metavar="original_image_path",type=str,help="Path of original medcines's package image")
    my_parser.add_argument("MUT_img_path",metavar="MUT_image_path",type=str,help="Path of MUT's package image")

    args = my_parser.parse_args()

    verify(args.original_img_path,args.MUT_img_path,args.model_path)

if __name__=="__main__":
    main()


    
