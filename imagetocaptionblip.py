#!/usr/bin/env python
# coding: utf-8

# In[2]:


from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import matplotlib.pyplot as plt


captioning_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
captioning_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_image_caption(image_file_path):
   
    try:
        image = Image.open(image_file_path).convert("RGB")
        
        processed_inputs = captioning_processor(images=image, return_tensors="pt")
        
        generated_output = captioning_model.generate(**processed_inputs, max_new_tokens=50)
        
        caption = captioning_processor.decode(generated_output[0], skip_special_tokens=True)
        
        return image, caption
    
    except Exception as e:
        # If error will come then it will show that error and return nothing 
        print(f"Error generating caption: {e}")
        return None, None

def display_image_with_caption(image_file_path):
   
    # we are calling the above function we made which return the original image and caption
    image, caption = generate_image_caption(image_file_path)
    
   
    if image and caption: # if image and caption both are there then only it will execute
        
        plt.figure(figsize=(8, 6)) # using for the size of image which we are showing
        
        
        plt.imshow(image)# used to fit the image in the plot
        
       
        plt.axis('off')# using for removing x,y values
        
        
        plt.title(caption, fontsize=16, wrap=True)#generated caption will be setted as title 
        
        
        plt.show()#it will show the image and caption both together
    else:  # if there is a error or no image or caption generated then it will be executed
        
        print("Unable to display image and caption due to an error in the processing.")

def main():
    
    image_file_path = r"enter the path of image" # path of the image for which we want caption
    
    
    display_image_with_caption(image_file_path)#calling the function which we made for showing image and caption

if __name__ == "__main__":
   
    main()













