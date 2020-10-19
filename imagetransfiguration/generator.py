from skimage.draw import disk
import random
random.seed(0)

class DummyCircles:
    def __call__(self, *args, **kwargs):
        n_circles, max_radius = kwargs['n_circles'], kwargs['max_radius']
        width, height = kwargs['width'], kwargs['height']
        circles = []
        for n in range(n_circles):
            rr, cc = disk((random.randint(width//4, width*3//4), random.randint(height//4, height*3//4)),
                          random.randint(1, max_radius))
            rr[rr > width] == width
            cc[cc > height] == height
            circles.append([rr, cc])
        return circles
