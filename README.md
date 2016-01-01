# Fish
A simulation of fish and their struggle to become smart. Written in Python and displayed with Pygame.

Contents:	
The code classes for neurons, a neural network (in the form of Brain), the fish, and the evolution process. Of these classes, only Evolution is used front-end, the rest serve to make the program work. 

How to use:		
After initializing an Evolution instance, call Evolution.tick() to update the screen. It is advisable to use this in a while loop to see smooth framerate refreshing.

Keyboard Shortcuts:		
Pressing 'p' will pause the simulation, 'b' will draw the brain of the longest living fish, and 'f' will turn off visuals while evolving in the background. 
