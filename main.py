from cppn_model import Cppn
from auto_encoder_model import Autoencoder
from typing import Tuple, Union
import math
import torch
import numpy as np
from matplotlib import pyplot as plt
import loader as ld
from collections import deque
import random
import copy
import time
from functools import reduce

NEURONS_IN_LAYER = 50
NUMBER_OF_LAYERS = 20
MUTATION_STRENGTH = 0.2
MAX_WEIGHT = 1000
FITNESS_DECAY = 0.8
DISPLAY_RESOLUTION = (940, 480)
POP_SIZE = 100
ELITISM = 0.5
SAVE_IMAGE_PERIOD = 1
HIDDEN_FUNCTION = torch.nn.Tanh
OUTPUT_FUNCTION = torch.nn.Sigmoid


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


class PopMember:
    def __init__(self, network):
        self.fitness = 0
        self.net = network

    def create_offspring(self):
        offspr = Cppn(NUMBER_OF_LAYERS, NEURONS_IN_LAYER,
                      HIDDEN_FUNCTION, OUTPUT_FUNCTION).cuda()
        offspr.load_state_dict(copy.deepcopy(self.net.state_dict()))

        offspr_params = list(offspr.parameters())
        params = list(self.net.parameters())

        for i in range(len(params)):
            d_mut = (torch.rand(offspr_params[i].data.shape) * 2 - 1).cuda()
            to_add = (d_mut/(params[i].grad.abs()+1)) * MUTATION_STRENGTH
            offspr_params[i].data += to_add
            offspr_params[i].data = offspr_params[i].data.clamp(
                -MAX_WEIGHT, MAX_WEIGHT)

        mem = PopMember(offspr)
        mem.fitness = self.fitness * FITNESS_DECAY
        return mem

    def __str__(self):
        return "Fitness: " + str(self.fitness)


@static_vars(CACHED_INPUTS_TO_NETWORK=dict())
def get_input_matrix_for_cppn(size: Tuple[int, int]):
    cache = get_input_matrix_for_cppn.CACHED_INPUTS_TO_NETWORK
    if size not in cache.keys():
        new_val = torch.empty(size[0] * size[1], 3)
        div = math.sqrt(2)
        for y in range(size[1]):
            for x in range(size[0]):
                xx = (x / (size[0] - 1)) * 2 - 1
                yy = (y / (size[1] - 1)) * 2 - 1
                rr = math.sqrt(xx ** 2 + yy ** 2) / div
                new_val[size[0] * y + x, 0] = xx
                new_val[size[0] * y + x, 1] = yy
                new_val[size[0] * y + x, 2] = rr
        new_val = new_val.cuda()
        cache.update({size: new_val})

    return cache[size]


def generate_image(model: Cppn, size: Tuple[int, int]):
    inp = get_input_matrix_for_cppn(size)

    output = model(inp)

    model.zero_grad()

    loss = (output ** 2).sum().sqrt()
    loss.backward()

    data = output.data.clamp(0, 1)

    data = data.reshape((size[1], size[0], 3))

    return data


def distance(latent1: torch.Tensor, latent2: torch.Tensor):
    # ret = ((latent1 - latent2) ** 2).sum() # euclidean
    ret = (latent1 - latent2).abs().sum()  # manhattan
    return ret


def get_latent(autoencoder: Autoencoder, image):
    # from 32x32x3 to 3x32x32
    image = image.transpose(2, 0).transpose(1, 2)

    image.unsqueeze_(0)
    lat = autoencoder.encoder(image).data.reshape((64,))
    return lat


def main():
    plt.ion()

    autoencoder = Autoencoder().cuda()
    autoencoder.load_state_dict(torch.load("autoencoder_save.pth"))
    autoencoder.eval()

    size = (32, 32)

    population = [PopMember(Cppn(NUMBER_OF_LAYERS, NEURONS_IN_LAYER,
                                 HIDDEN_FUNCTION, OUTPUT_FUNCTION).cuda())
                  for i in range(POP_SIZE)]
    explored_latent = deque(maxlen=10)
    explored_latent.append(get_latent(
        autoencoder, generate_image(population[0].net, size)))

    for generation in range(1000):
        time_beg = time.time()
        # assign fitness
        for member in population:
            mn = min(map(lambda latent: distance(get_latent(
                autoencoder, generate_image(member.net, size)), latent), explored_latent))
            member.fitness = (mn + member.fitness) / 2

        # sort and prune the population
        population.sort(key=lambda x: x.fitness, reverse=True)
        population = population[:int(POP_SIZE * ELITISM)]

        explored_latent.append(get_latent(
            autoencoder, generate_image(population[0].net, size)))

        # repopulate population
        ln = len(population) - 1
        for _ in range(int(POP_SIZE * (1 - ELITISM))):
            population.append(
                population[random.randint(0, ln)].create_offspring())

        # log
        print("Generation: " + str(generation) +
              "; time elapsed: " + str(time.time() - time_beg))
        if generation % SAVE_IMAGE_PERIOD == 0:
            time_gen = time.time()
            # create and show leader image
            image = generate_image(
                population[0].net, DISPLAY_RESOLUTION).cpu().numpy()
            image = (image * 255).astype(np.uint8)

            ld.save_img(image, "generated images/computer art " +
                        str(generation) + " generation" + str(time.time()))

            print("Time spent on cool high res image generation: " +
                  str(time.time() - time_gen))


if __name__ == "__main__":
    main()
