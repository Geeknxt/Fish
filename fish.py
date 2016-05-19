# By Richard Zhou
# My attempt at trying to understand neural networks and optimizing them
# with genetic algorithms by making some fish.

import random
import math
import pygame
import sys
import copy


class Neuron:
    """A single neuron with two arrays for weights and inputs that fires
    based upon a sigmoid curve instead of a threshold."""
    # for threshold, add a paramater for threshold value and convert
    # evalOutput to a bool function that checks if sigmoid > threshold

    def __init__(self, numInputs, weights=[]):
        # this is just for housekeeping
        self.numInputs = numInputs
        self.inputs = []  # this is the input values
        # initially zero inputs
        if not self.inputs:
            for i in range(0, self.numInputs):
                self.inputs.append(0)

        # this is how much each input affects the neuron
        self.weights = []
        if weights:
            self.weights = list(weights)
        if not self.weights:
            for i in range(0, self.numInputs):
                # just give it a random weight
                self.weights.append(random.uniform(-1, 1))

    def __str__(self):
        # print its information
        print("Number of Inputs/Weights: " + str(self.numInputs))
        print("Weights:", end=' ')
        for i in self.weights:
            print(str(i), end=' ')
        return "\n"

    def __repr__(self):
        info = ""
        for i in self.weights:
            info += (str(i) + ' ')
        return info

    def giveInputs(self, inputs):
        """Give the neuron a set of inputs."""
        self.inputs = list(inputs)

    def getWeights(self):
        """Returns the neuron's weights for its inputs."""
        return list(self.weights)

    def evalOutput(self):
        """Returns the weighted sigmoid sum of the inputs."""
        axonVal = 0
        # add up each weighted input and run it through sigmoid
        for i in range(0, self.numInputs):
            axonVal += (self.inputs[i] * self.weights[i])
        axonVal = 1 / (1 + math.exp(-axonVal))
        return axonVal

###############################################################################


class Brain:
    """A basic neural network that optimizes based on genetic algorithms
    and not through backpropagation. Used for the brains."""

    def __init__(self, numNeurons, density, brain=[], connectome=[]):
        self.numNeurons = numNeurons
        self.density = density   # this is # synapses per neuron
        self.brain = list(brain)   # this is for the neurons that hold the weights
        self.signals = []   # this is for lookup of the last output
                            # 8 larger because 3 are for senses (2 eyes and
                            # hunger) and 5 are "bias" neurons
        self.connectome = list(connectome)  # this is for the neural "map"
        self.numSenses = 3  # number of inputs
        self.numBias = 5  # number of bias neurons

        # initialization
        # setting up the brain
        if not self.brain:
            for i in range(0, self.numNeurons):
                newron = Neuron(self.density)
                self.brain.append(newron)

        # randomize the connectome
        if not self.connectome:
            # for each neuron
            for i in range(0, self.numNeurons):
                # ake a random list of dendrites
                connections = []
                for j in range(0, density):
                    connections.append(random.randint(0, self.numNeurons + self.numSenses + self.numBias - 1))
                self.connectome.append(connections)

        # "zero" the signals array
        for i in range(0, self.numNeurons + self.numSenses + self.numBias):
            self.signals.append(1)

    def __repr__(self):
        return str(self.brain)

    def update(self, senses):
        """This updates the state of all the neurons by taking inputs from the
        eyes (and other senses) and runs it through the neural network
        (the brain), which returns instructions for movement."""
        self.senses = list(senses)  # for now it is [eye1, eye2, hunger]
        self.numSenses = len(self.senses)
        self.senses[-1] -= 100  # this is for "balancing out" the hunger
                                # as 100 is the optimal hunger value

        # updating "sensory neurons"
        for i in range(0, self.numSenses):
            self.signals[i] = self.senses[i]

        # pad for bias neurons
        for i in range(self.numSenses + self.numBias, self.numNeurons + self.numSenses + self.numBias):
            # this will be for each neuron
            tempIn = []  # this is each neuron's input
            for k in range(0, self.density):
                # this is the last output of each neuron's connectome connection
                tempIn.append(self.signals[self.connectome[i - (self.numSenses + self.numBias)][k]])
            self.brain[i - (self.numSenses + self.numBias)].giveInputs(tempIn)
            # update with each neuron's output
            self.signals[i] = self.brain[i - (self.numSenses + self.numBias)].evalOutput()

        # this is for returning the movement options [fw, cw, ccw]
        moveOut = []
        for i in range(1, 4):
            moveOut.append(self.signals[-i])
        return moveOut

    def breed(self):
        """For now, this asexually clones itself with a high rate of
        mutation to encourage genetic diversity."""
        self.mutationRate = 0.2
        self.mutationSeverity = 0.3
        childBrain = []
        childConnectome = list(self.connectome)

        # clone the weights with mutations
        for i in self.brain:
            clone = list(i.getWeights())
            for j in range(0, len(clone)):
                if random.random() < self.mutationRate:
                    if random.getrandbits(1):
                        clone[j] += self.mutationSeverity
                    else:
                        clone[j] -= self.mutationSeverity

                    # make sure not crazy
                    if clone[j] > 1:
                        clone[j] -= 2 * self.mutationSeverity
                    elif clone[j] < -1:
                        clone[j] += 2 * self.mutationSeverity
            childBrain.append(Neuron(self.density, clone))

        # clone the connectome with mutations
        for i in childConnectome:
            for j in i:
                if random.random() < self.mutationRate:
                    j = random.randint(0, self.numNeurons + self.numSenses + self.numBias - 1)

        # create our new spawn of satan
        child = Brain(self.numNeurons, self.density, childBrain, childConnectome)
        return child

###########################################################################################


class Fish:
    """A fish that has a neural network brain with two eyes, and can move and rotate."""

    # parameters are for (BRAIN)(POSITION/ANGLE)(INHERIT)
    def __init__(self, numNeurons, density, posX, posY, angle, brain=0):
        self.numNeurons = numNeurons
        self.density = density
        self.brain = copy.deepcopy(brain)
        self.eyes = [0, 0, 0]  # this is what the eye sees + HUNGER
        self.position = list([posX, posY, angle])
        self.hunger = 100
        self.lifespan = 1

        # initialize a brain
        if not brain:
            self.brain = Brain(self.numNeurons, self.density)

    def getPosition(self):
        """Returns the position of the fish as list containing the x and y coordinates
        as well as the angle from standard position that it is facing, as integers."""
        pos = list(self.position)
        for i in range(0, len(pos)):
            pos[i] = int(pos[i])
        return pos

    def giveSight(self, sight):
        """Used to update the values of what the fish sees."""
        # sight should be a triple [s1, s2, hunger], do not normalize hunger yet
        self.eyes = list(sight)

    def getMove(self):
        """Calculates its movement (forwards and rotational) based on what it sees
        and moves itself accordingly."""
        # should return movement as triple [fw, cw, ccw]
        twitch = list(self.brain.update(self.eyes))
        # update the angle its facing based on ANGULAR ROTATION RATE
        self.position[-1] = (12 * (twitch[-1] - twitch[-2]) + self.position[-1] + 360) % 360
        # update x and y based on SPEED RATE and RESOLUTION
        self.position[0] = (self.position[0] + 8 * (twitch[0] * math.cos(math.radians(self.position[-1])))) % 1280
        self.position[1] = (self.position[1] - 8 * (twitch[0] * math.sin(math.radians(self.position[-1])))) % 720

    def updateHunger(self, amt):
        """Updates the hunger value based on starvation and feeding."""
        self.hunger += amt

    def getHunger(self):
        """Tells us how hungry it is. Can die from overeating or undereating."""
        return int(self.hunger)

    def updateLife(self):
        """Literally just increases the lifespan by a tick."""
        self.lifespan += 1

    def getLife(self):
        """Returns how long this fish has been living."""
        return int(self.lifespan)

    def breedFish(self):
        """Creates a new fish with a similar brain structure."""
        return Fish(self.numNeurons, self.density, random.randint(0, 1279), random.randint(0, 719),
                    random.randint(0, 359), self.brain.breed())

###########################################################################################


class Evolution:
    """A simulation of fish with neural network brains. The best fish in the pond evolve
    and eventually, the pond's fish population as a whole evolves."""

    def __init__(self, numFood, numFish):
        # PYGAME STUFF
        pygame.init()
        self.scrSize = self.width, self.height = 1280, 720  # set up screen size
        self.screen = pygame.display.set_mode(self.scrSize)  # make a screen
        pygame.display.set_caption("Fish Evolution with Neural Networks")  # sets caption

        # trying to get a white background for blitting later
        self.background = pygame.Surface(self.scrSize)
        self.background = self.background.convert()
        self.background.fill((255, 255, 255))
        self.screen.blit(self.background, (0, 0))

        # just for settings to pause and draw brains and speed up
        self.paused = False
        self.drawNN = False
        self.visuals = True

        # ACTUAL EVOLUTION STUFF
        self.numFood = numFood    # amount of food in the pond
        self.numFish = numFish    # number of fish in the pond
        self.pond = []  # array of fish
        self.food = []  # array of food locations (position in x,y)
        self.foodLoss = 0.2

        for i in range(0, self.numFish):
            # FISH STATS GO HERE
            self.pond.append(Fish(15, 4, random.randint(0, self.width - 1), random.randint(0, self.height - 1), random.randint(0, 359)))
        # add and draw food
        for i in range(0, self.numFood):
            self.food.append(list([random.randint(0, self.width - 1), random.randint(0, self.height - 1)]))
            # here we choose to use green and make the radius 10 pixels
            pygame.draw.circle(self.screen, (0, 175, 0), self.food[-1], 10)

        # update the screen
        pygame.display.update()

    def tick(self):
        """This is the function that we update every tick to make stuff happen."""
        # check for inputs for exit and pause and brain and such
        for event in pygame.event.get():
            # check for exit
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.KEYDOWN:
                # check to see if need to draw or pause
                if event.key == pygame.K_p:
                    self.paused = not self.paused
                if event.key == pygame.K_b:
                    self.drawNN = not self.drawNN
                if event.key == pygame.K_f:
                    self.visuals = not self.visuals

        # if we paused just skip this tick
        if self.paused:
            return

        # PSEUDOCODE FOR BELOW
        # redraw
        # for each fish
            # subtract hunger
            # for each food
                # check if eating
                    # update hunger/alive, update food & blit new/old
                # update eyevalues
            # IF STILL ALIVE give senses and update its position
                #  if not alive replace it
            # blit fish

        # redraw food
        # PUSH DRAWINGS

        # redraw screen as white
        self.screen.blit(self.background, (0, 0))

        # For Each Fish
        for i in range(0, self.numFish):
            self.pond[i].updateHunger(-0.34)  # make it starve a little
            self.pond[i].updateLife()   # update its lifespan
            senses = [0, 0, 0]  # zero out the senses

            # FISH POSITION INFO
            fishPos = list(self.pond[i].getPosition())
            # calculate the location of the eyes if they are 180 apart and 8 units away from brain
            # as (X1,Y1)(X2,Y2)
            eyePos = [[int(fishPos[0] + 8 * math.cos(math.radians((fishPos[-1] + 90) % 360))),
                       int(fishPos[1] - 8 * math.sin(math.radians((fishPos[-1] + 90) % 360)))],
                      [int(fishPos[0] + 8 * math.cos(math.radians((fishPos[-1] + 270) % 360))),
                       int(fishPos[1] - 8 * math.sin(math.radians((fishPos[-1] + 270) % 360)))]]

            # for each food
            for j in range(0, self.numFood):
                # check if we ate the food
                if ((fishPos[0] - self.food[j][0])**2 + (fishPos[1] - self.food[j][1])**2) < 225:
                    # if we ate food fill hunger by 45 HUNGER POINTS and update the food array
                    self.pond[i].updateHunger(45)
                    self.food.pop(j)
                    j -= 1
                    # add another food randomly on the map
                    self.food.append(list([random.randint(0, self.width - 1), random.randint(0, self.height - 1)]))
                # if we didn't eat food
                else:
                    # update each eye vaule
                    for k in range(0, 2):
                        # add the result of the function f(d)=1/(d+0.25)**2 where d is distance
                        senses[k] += 1 / (math.sqrt((eyePos[k][0] - self.food[j][0])**2 + (eyePos[k][1] - self.food[j][1])**2) + 0.25)**2

            # update the hunger value
            senses[-1] = self.pond[i].getHunger()

            # if our fish is dead
            if senses[-1] > 200 or senses[-1] < 0:
                self.pond.pop(i)
                i -= 1

                # pick probabilistically based on f(x) 1/(x+1.5) where x is the indices of the best fitness
                # so the more fit are more likely to get populated
                sexPool = sorted(self.pond, key=Fish.getLife, reverse=True)
                for k in range(0, len(sexPool) // 2):
                    if random.random() < 1 / (k + 1.5):
                        # add the new mutated fish to pond
                        self.pond.append(sexPool[k].breedFish())
                        break
                # if its too unfit we just rig it
                if len(self.pond) != self.numFish:
                    self.pond.append(sexPool[0].breedFish())

            # if our fish is alive then update its senses and position
            else:
                self.pond[i].giveSight(senses)
                self.pond[i].getMove()
                fishPos = list(self.pond[i].getPosition())
                eyePos = [[int(fishPos[0] + 8 * math.cos(math.radians((fishPos[-1] + 90) % 360))),
                           int(fishPos[1] - 8 * math.sin(math.radians((fishPos[-1] + 90) % 360)))],
                          [int(fishPos[0] + 8 * math.cos(math.radians((fishPos[-1] + 270) % 360))),
                           int(fishPos[1] - 8 * math.sin(math.radians((fishPos[-1] + 270) % 360)))]]
                # check if this fish needs to be highlighted and/or drawn
                if self.visuals:
                    self.drawFish(fishPos, eyePos, (max(self.pond, key=Fish.getLife) == self.pond[i]))

        # draw food if we should
        if self.visuals:
            for i in range(0, self.numFood):
                # redraw each food
                pygame.draw.circle(self.screen, (0, 175, 0), (self.food[i][0], self.food[i][1]), 10)

        # Draw brain if we should
        if self.drawNN and self.visuals:
            self.drawBrain(max(self.pond, key=Fish.getLife).brain)

        # push updates to screen
        if self.visuals:
            pygame.display.update()

    def drawFish(self, location, eyeLoc, highlight):
        """Draws the fish given by its location and orientation."""
        # drawing the body with width 8 and length 20
        pygame.draw.line(self.screen, (175, 175, 0), (location[0], location[1]),
                         (location[0] - int(20 * math.cos(math.radians(location[-1]))),
                          location[1] + int(20 * math.sin(math.radians(location[-1])))), 8)

        # drawing head with radius 8
        if not highlight:
            pygame.draw.circle(self.screen, (175, 175, 0), (location[0], location[1]), 8)
        elif highlight:
            # draw their head in light blue if they are the best
            pygame.draw.circle(self.screen, (0, 200, 200), (location[0], location[1]), 8)

        # draw eyes with radius 2
        for l in eyeLoc:
            pygame.draw.circle(self.screen, (0, 0, 0), (l[0], l[1]), 2)

    def drawBrain(self, brains):
        """Draws the brain of a given fish in the corner, with red symbolizing a
        positive weight and blue symbolizing a negative weight."""
        # We're going to assume that this brain has 15 Neurons and 4 Synapses per Neuron
        # and the fish takes 3 senses and outputs 3 as well (not using brains' args)
        displace = (1000, 200)  # where the drawing is centered
        neuronPos = []   # where each neuron is, including eyes and bias like in signal

        # initialize where each appendage is (with neurons having a radius of 30)
        neuronPos.append([displace[0] - 170, displace[1]])  # eye 1
        neuronPos.append([displace[0] + 170, displace[1]])  # eye 2
        for i in range(0, brains.numBias + 1):
            neuronPos.append([displace[0] + int(60 * math.cos(math.radians(i * (360 / brains.numBias)))),
                              displace[1] - int(60 * math.sin(math.radians(i * (360 / brains.numBias))))])
        for i in range(0, brains.numNeurons - 3):
            neuronPos.append([displace[0] + int(100 * math.cos(math.radians(i * (360 / (brains.numNeurons - 3))))),
                              displace[1] - int(100 * math.sin(math.radians(i * (360 / (brains.numNeurons - 3)))))])
        neuronPos.append([displace[0], displace[1] + 350])  # fw
        neuronPos.append([displace[0] + 100, displace[1] + 350])  # cw
        neuronPos.append([displace[0] - 100, displace[1] + 350])  # ccw

        # start drawing the connectome
        for i in range(0, len(brains.connectome)):  # for each neuron
            redBlue = list(brains.brain[i].getWeights())  # list of our current neuron's weights
            # draw the colored line for each input
            for k in range(0, len(redBlue)):
                pygame.draw.line(self.screen, (int(127.5 + 127.5 * redBlue[k]), 0, int(127.5 - 127.5 * redBlue[k])),
                                 neuronPos[i + brains.numSenses + brains.numBias], neuronPos[brains.connectome[i][k]], 6)
        # draw the nodes
        for i in neuronPos:
            pygame.draw.circle(self.screen, (25, 25, 25), i, 15)

################################################################################


# #-----TESTING HERE------# #

e = Evolution(20, 30)
while True:
    e.tick()
