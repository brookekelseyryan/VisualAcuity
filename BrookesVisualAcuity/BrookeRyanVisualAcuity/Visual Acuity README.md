# Visual Acuity README

The code can be viewed on GitHub here: https://github.com/brookekelseyryan/VisualAcuity

I wrote quite a bit of code for this project, so I'll summarize here the module structure that delineates what to find in the .zip file. All of the code included was written by me, and I did of course leverage Deep Learning libraries like tensorflow, keras, etc. 

* **config**: contains the YAML files with the training parameters 
* **data**: contains a helper class Dataset that I wrote which loads in the pickled processed image data
* **model**: contains code I wrote for the VGG16 transfer model
* **preprocessing**: this is all of the code used to preprocess and categorize all of the image data from the original dataset 
* **Teller**: sub-module of preprocessing, special processing that was applied to Teller acuities. 
* **util**: contains a utility method to connect my runs to Weights and Biases to produce visualizations. 
* **main.py** entry point to the program

