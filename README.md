

<p align="center">
<img  src="https://i.ibb.co/27bfXmn/Sm3ni-logos-transparent.png" width="320" align="center" alt="Sm3ni Logo"  />

[![GitHub forks](https://img.shields.io/github/forks/nadaabdelmaboud/Sm3ni)](https://github.com/nadaabdelmaboud/Sm3ni/network/members) [![GitHub Repo stars](https://img.shields.io/github/stars/nadaabdelmaboud/Sm3ni)](https://github.com/nadaabdelmaboud/Sm3ni/stargazers) [![GitHub contributors](https://img.shields.io/github/contributors/nadaabdelmaboud/Sm3ni)](https://github.com/nadaabdelmaboud/Sm3ni/graphs/contributors) 
![GitHub](https://img.shields.io/github/license/nadaabdelmaboud/Sm3ni)[
![GitHub issues](https://img.shields.io/github/issues/nadaabdelmaboud/Sm3ni)](https://github.com/nadaabdelmaboud/Sm3ni/issues)
</p>


### Table of Contents

1. [Description](#Description)
2. [Pipeline](#Pipeline)
	* [Preprocessing](#Preprocessing)
		* [Smoothing](#Smoothing)
		* [Illumination](#Illumination)
		* [Binarization](#Binarization)
		* [Deskewing](#Deskewing)
	* [Staff Lines Removal](#Staff-Lines-Removal)
		* [Horizontal Staff-Lines Removal](#Horizontal-Staff-Lines-Removal)
		* [Non Horizontal Staff-Lines Removal](#Non-Horizontal-Staff-Lines-Removal)
	* [Segmentation](#Segmentation)
	* [Features Extraction](#Features-Extraction)
		* [Stems](#Stems)
		* [Notes Heads](#Notes-Heads)
		* [Beams](#Beams)
		* [Up or Down](#UporDown)
		* [Flags](#Flags)
	* [Classification](#Classification)
	* [Generating Output](#Generating-Output)
		* [Text Output](#Sample-Text-Output)
		* [Audio Output](#Sample-Audio-Output)
5. [Run](#Run)
6. [Contributing](#Contributing)
7. [Contributors](#Stay-in-touch)
8. [License](#License)

## Description

**Sm3ni** is an Optical Musical Recognition project written in python that converts music sheets images to a text file representing the musical notes then to a .wav audio file that represents the music sheet .

This repo will contain our used pipeline for the OMR based on different cases for the input images . 
## Pipeline

* ## Preprocessing

	![Preprocessing](https://i.ibb.co/0Qw2XZx/Untitled-Diagram-Page-7-1.png)
	* ### Smoothing 

		>  We first apply a smoothing test by calculating the signal to noise ratio in the image .
		
		> If the **SNR** is above some threshold the **bilateral smoothing filter** is applied on the image

	* ### Illumination
		> Uneven illumination test is applied on the image
		
		> If the image is unevenly illuminated a contrast enhancement algorithm called **screened poisson contrast enhancement**  is applied on the image
	
	* ### Binarization
		> If the image is unevenly illuminated we apply **Feng Local Thresholding** for low quality documented images.
		
		> If the image is evenly illuminated we apply **OTSU Local Thresholding** .
		
	* ### Deskewing

		> Rotate the image from 0 to 360 degree then find the angle of maximum horizontal projection sum and rotate the image with the right angle .
	
		![Preprocessing Outputs](https://i.ibb.co/jkkWt9j/Untitled-Diagram-Page-1.png)

* ## Staff-Lines Removal

	![Staff-Removal](https://i.ibb.co/BnbJT6j/Untitled-Diagram-Page-2-1.png)

	> Horizontal/Non-horizontal test is first applied on the image then 2 different techniques are applied based on the test.
	* ### Horizontal Staff-Lines Removal
		> Get **horizontal projection** of the image  
		
		> Find the peaks and get the staff-height and staff-space
		
		> Remove lines with width same as the peaks' width
	* ### Non-Horizontal Staff-Lines Removal
		> Apply **run length encoding** .
		
		> Find the mod of consecutive black pixels to be the staff height and mod of consecutive white pixels to be the staff-space.
		
		> Remove all regions with height more than the staff height.
		
		> Get the removed lines image by subtracting the removed symbols image from the original image.
		

	![Staff-Removal Output](https://i.ibb.co/F6Kqv4V/Untitled-Diagram-Page-3.png)
* ## Segmentation
	![Segmentation](https://i.ibb.co/1zT175S/Untitled-Diagram-Page-4.png)
	> Segmenting the image into staff-segments.
	
	> Segmenting each staff segment to the musical symbols.
	
	> Detect **Clef** position and rotate the whole image 180 degree if it exists in the bottom right corner.
	
	> For non horizontal images **de-skew** each symbol from -45 to 45 degree till getting the right angle for rotation from projection sum . 
	
	
* ## Features Extraction
	> From this step we get a feature vector describing each symbol that will be used later in the classification . 
	
	* ### Stems
		> Detect stems in the symbol by vertical projection then remove them.
		
	* ### Notes-Heads
		> Get all regions left in the symbol and threshold on each region by **solidity**,**eccentricity**,**area**,**width** and **height** with respect to the **staff-space**
		
		> count all valid heads to be number of black heads .
		
		> Apply **region filling** to the symbol then repeat the previous steps.
		
		> count all valid heads to be number of white heads.

	* ### Beams
		> Get all regions left in the symbol then apply thresholding by **aspect ratio** to detect number of beams lines.
		
	* ### UporDown
		> Detect whether notes is up or down by comparing the center of the highest head with the tallest stem in the symbol .
		
	* ### Flags
		> Apply **skeletonization** to the symbol.
		
		> Find all **connected right down/left up paths** with thresholding on the path's length.
		
		> Count number of valid flags.
		
* ## Classification
	![Classification](https://i.ibb.co/H2Zr2ff/Untitled-Diagram-Page-5-1.png)
* ## Generating Output
	![Output](https://i.ibb.co/94cLYrR/Untitled-Diagram-Page-6.png)
	* #### Sample Text Output

		![Text](https://i.ibb.co/zrbHF7R/Screenshot-from-2021-01-28-19-58-57.png)	
	* #### Sample Audio Output
		[Audio.wav](https://drive.google.com/file/d/13Y3azkBrkqGE6kC38r5cLsaKzd7YfgR7/view?usp=sharing)

## Tools
* Python
* Numpy
* Skimage
* Opencv
* Os : Linux-Ubuntu 		
		
## Run
```bash
$ conda env create -f requirements.yml
$ conda activate omrproject
$ python src/main.py <Input Folder Absolute Path> <Output Folder Absolute Path>
```
 
## Contributing
	
```bash
1. Fork this repo
2. Create new branch
	$ git checkout -b <YourBranch>
3. Add your modifications then
	$ git commit -m "Commit Message"
	$ git push origin <YourBranch>
4. Create PR
```
## Stay in touch

- [Nihal Mansour](https://github.com/Nihal-Mansour)
- [Hager Ismael](https://github.com/hagerali99)
- [Menna Mahmoud](https://github.com/MENNA123MAHMOUD)
- [Nada AbdElmaboud](https://github.com/nadaabdelmaboud)

## License

**Sm3ni** is [MIT licensed](LICENSE).

