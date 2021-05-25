# Visual Acuity Project



*sharpness of vision, measured by the ability to discern letters or numbers at a given distance according to a fixed standard.*

An eye chart, or **optotype**, is a chart used to measure visual acuity. Eye charts are often used by health care professionals, such as optometrists, physicians or nurses, to screen persons for vision impairment.

The idea is to study how the recognition performance of NNs decreases with blurring and compare it to humans, with the goal of coming up with better tests.



## Questions

RE the Goal slide on page 4:

- The algorithm will be trained on **large** characters with **low** optical distortion 
  - Does it have to just be large? Can we not also use the small and medium sizes? (does it matter what size of character it is trained on?)
  - Would it just be trained on these particular character/fonts that we have in here?  



What kind of better tests? Better vision tests? 



Would the tests be among these kinds of symbols again?



Would it be for like iphone progressive blurriness kind of test? 







Sphere - Magnifying glass 

Cyl - Curvature that is not the same in both directions, like a football 



Everything below 5 is "easy"? 



How much distortion can it tolerate before it gets the wrong answer 





softmax for patient recognizing the rotated distortion 



CNN with softmax at the top 

couple different experiments 



mimic different types of humans 



"Acuity" rather than recognition 



Tiers of experiments:

1. lowest levels of optotpe and 
2. all fonts 



a month a 


# Notes, Week of May 17
what if more training data was taken from like Mnist or something like that?
is there a dataset for like non-handwritten stuff?
idk what results he's going for but there could be a correlation between literacy and vision tests
maybe to frame the question it'd be something like, is there a correlation between symbol recognition and vision tests
idk what people he's testing the app on
personally I feel like the symbol recognition, at least for humans, is harder
idk what kind of standardized like, training there could be for that
symbols are so ingrained in your head
maybe like facial visual acuity tests? idk