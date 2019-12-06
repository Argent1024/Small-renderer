# The complete code is also available at https://github.com/Argent1024/Small-renderer/blob/master/render.py
# This is the main part of a simplified path-tracing renderer. Including intersection of shapes, path tracing alg.
# There is another file that contains the common materials, utilizes and sampler functions for MC Integration.
# Change sample number and image size to get the result quickly (since this program uses only one thread).
#
# I chose this simply because I like computer graphics and want to do related works in the future.
# And since I'm applying for this CS/DA program, which focuses more on the CG area, the first thing that
# came into my mind was to write the path-tracing algorithm. I hope writing something related to CG can
# show my knowledge and efforts in this area.

import numpy as np
from PIL import Image
from material import normalize, RedGlass, RedWall, BlueWall, YellowGlass, DefaultMat, WhiteGlass, Mirror

class Ray:
    def __init__(self, p, d, tmin=0., tmax=np.finfo('f').max):
        self.dir, self.p, self.tmin, self.tmax, = normalize(d), p, tmin, tmax

class Plane:    # A hard code plane, center can only be (x,0,0), (0,y,0), (0,0,z)
    def __init__(self, c, mat=DefaultMat, length=50):  # which axises are not zero
        self.length, self.c, self.mat = length, c, mat
        self.index = np.nonzero(c)[0][0]
        self.zero_axis = np.where(c == 0)[0]
        self.n = normalize(-c)

    def intersect(self, ray):  # intersect of ray and plane, return intersection point
        if ray.dir[self.index] != 0:
            length = (self.c[self.index] - ray.p[self.index]) / ray.dir[self.index]
            p = ray.p + length * ray.dir
            # check intersection point not too close to origin and fall on plane
            if ray.tmin < length < ray.tmax and all(np.abs(p[self.zero_axis]) <= np.array([self.length, self.length])):
                ray.t = length
                return {"p": p, "n": self.n, "material": self.mat}
        return None

class Sphere:
    def __init__(self, c, material, radius):
        self.center, self.r, self.material = c, radius, material

    def intersect(self, ray):    # intersect of ray and sphere, return intersection point
        p2c = self.center - ray.p
        length = np.dot(p2c, ray.dir)
        distance = np.sqrt(np.abs(np.dot(p2c, p2c) - np.power(length, 2)))
        if self.r < distance or (length < 0 and self.r < np.linalg.norm(p2c)):
            return None
        q = np.sqrt(np.power(self.r, 2) - np.power(distance, 2))
        t = min(np.abs(length - q), np.abs(length + q))
        if t < ray.tmin:  # Ignore intersection too close to origin, avoid rounding error
            t = max(np.abs(length - q), np.abs(length + q))
        if t > ray.tmax:  # See if t smaller than max distance
            return None
        ray.tmax = t
        isect_p = ray.p + t * ray.dir
        return {"p": isect_p, "n": normalize(isect_p - self.center), "material": self.material}

def find_hit(scene, ray):
    isect = None
    for primitive in scene:
        t_isect = primitive.intersect(ray)
        if t_isect:
            isect = t_isect
    return isect

def light(p, n, material, scene, emission, l_sample=4):     # light source is a plane locates on celling
    if emission and np.abs(p[0] - 50) <= 0.01 and all(np.abs(p[[1, 2]]) <= np.array([20, 20])):
        return np.array([1., 1., 1.])  # emit light if p actually locates on the light source
    illumination = np.zeros(3)
    # random sample and cast shadow ray
    for i in range(l_sample):
        light_point = np.array([50., 40 * np.random.random() - 20, 40 * np.random.random() - 20])
        dir, ray_len = normalize(light_point - p), np.linalg.norm(light_point - p)
        if not find_hit(scene, Ray(p + 0.1 * dir, dir, tmax=ray_len - 0.5)):   # See if shadow ray hit anything between
            illumination += np.abs(np.dot(dir, n)) * np.array([1., 1., 1.])
    return illumination * material.brdf() / l_sample

def trace_ray(ray, scene, max_iter):
    color, weight, emission = np.array([0., 0., 0.]), np.array([1., 1., 1.]), True  # init parameters
    for iter_num in range(max_iter):
        isect = find_hit(scene, ray)   # find the closet intersection point
        if not isect:                  # Break if hit nothing
            break
        mat = isect["material"]
        new_dir, p = mat.sample(-ray.dir, isect["n"])  # Generate new ray and store it's probability
        if not mat.is_delta():
            color += weight * light(isect["p"], isect["n"], isect["material"], scene, emission)   # Calculate light
            weight *= np.abs(np.dot(new_dir, isect["n"]))      # add cos term if material's brdf is not delta function
        weight *= mat.brdf() / p                               # add the probability term
        emission = mat.is_delta()                              # whether direct light should be include
        ray = Ray(isect["p"] + 0.1*new_dir, new_dir, tmin=0.11)  # Move the origin a little bit to avoid rounding error
    return color

if __name__ == "__main__":
    img, camera_p, sample_num, max_iter = Image.new('RGB', (500, 500)),  np.array([0., 0., -203.]), 128, 8
    my_scene = [Sphere(np.array([-30., -20., 20.]), Mirror, 20), Sphere(np.array([-40., -30., -20.]), YellowGlass, 10),
                Sphere(np.array([-40., -10., -30.]), RedGlass, 10), Sphere(np.array([-30., 20., -10.]), WhiteGlass, 20),
                Plane(np.array([50, 0, 0])), Plane(np.array([-50, 0, 0])), Plane(np.array([0, 0, 50])),
                Plane(np.array([0, 50, 0]), RedWall), Plane(np.array([0, -50, 0]), BlueWall)]
    # ray trace every pixel
    for h in range(img.height):
        for w in range(img.width):
            color = np.zeros(3)
            for sample in range(sample_num):
                canvas_point = np.array([1. - 2. * (h+np.random.random()) / img.height,
                                         2. * (w + np.random.random()) / img.width - 1., -200.])
                color += trace_ray(Ray(camera_p, canvas_point - camera_p), my_scene, max_iter)
            img.putpixel((w, h), tuple(int(255 * i / sample_num) for i in color))
    img.show()
    img.save("result_" + str(sample_num) + "_" + str(max_iter) + ".png")
