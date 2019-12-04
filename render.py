import numpy as np
from PIL import Image
normalize = lambda x: x / np.linalg.norm(x)
reflect = lambda in_dir, normal: 2. * np.dot(normal, in_dir) * normal - in_dir
def glass_sample(in_dir, normal, eta=0.666):
    cos = np.dot(in_dir, normal)
    if cos < 0:         # Ray from inside
        eta, cos, normal = 1 / eta, np.abs(cos), -normal
    r0 = np.power(eta - 1, 2) / np.power(eta + 1, 2)
    p = r0 + (1 - r0) * np.power(1 - cos, 5)   # Schlick's approximation
    if np.random.random() < p:
        return (reflect(in_dir, normal), 1)
    else:
        t = 1 - np.power(eta, 2) * (1 - np.power(cos, 2))
        return (-eta * in_dir + (eta * cos - np.sqrt(t)) * normal, 1) if t >= 0 else (reflect(in_dir, normal), 1)
def cosWeightHemiSPhere(in_dir, n):
    theta, cos_p = 2 * np.pi * np.random.random(), np.random.random()
    v = normalize(np.array([0, 0, 1]) - n[2] * n) if (n[2] != 1 and n[2] != -1) else np.array([1, 0, 0])
    sin_p, T = np.sqrt(1 - cos_p * cos_p), np.transpose(np.array([v, np.cross(v, n), n]))
    return np.matmul(T, np.array([np.sin(theta)*sin_p, np.cos(theta)*sin_p, cos_p])), cos_p / np.pi
Sampler = {"Diffuse": cosWeightHemiSPhere, "Glass":glass_sample, "Mirror": lambda i, n: (reflect(i, n), 1)}
Diffuse = {"type": "Diffuse", "brdf": np.array([0.9, 0.9, 0.9]) / np.pi, "isdelta": False}
Red, Blue = {"type": "Diffuse","brdf":np.array([0.9, 0.75, 0.75])/np.pi, "isdelta": False}, {"type":"Diffuse","brdf":np.array([0.75, 0.75, 0.9])/np.pi, "isdelta":False}
class Ray:
    def __init__(self, p, d, tmax=np.finfo('f').max, tmin=0.):
        self.dir, self.p, self.tmax, self.tmin = normalize(d), p, tmax, tmin
class Plane: # A hard code plane, center can only be (x,0,0), (0,y,0), (0,0,z)
    def __init__(self, length, center, normal, axis, material=Diffuse):  # which axises are not zero
        self.length, self.center, self.normal, self.zeroaxis, self.index, self.material \
            = length, center, normal, axis, {0, 1, 2}.difference(set(axis)).pop(), material
    def intersect(self, ray):  # intersect of ray and box, return t/f, isect point, normal
        if ray.dir[self.index] != 0:
            length = (self.center[self.index] - ray.p[self.index]) / ray.dir[self.index]
            p = ray.p + length * ray.dir
            if ray.tmin < length < ray.tmax and all(np.abs(p[self.zeroaxis]) <= np.array([self.length, self.length])):
                ray.t = length
                return {"p": p, "n": self.normal, "material": self.material}
        return None
class Sphere:
    def __init__(self, c, material, radius):
        self.center, self.r, self.material = c, radius, material
    def intersect(self, ray):    # intersect of ray and sphere, return t/f, isect point, normal
        p2c = self.center - ray.p
        length = np.dot(p2c, ray.dir)
        distance = np.sqrt(np.abs(np.dot(p2c, p2c) - np.power(length, 2)))
        if self.r < distance or (length < 0 and self.r < np.linalg.norm(p2c)):
            return None
        q = np.sqrt(np.power(self.r, 2) - np.power(distance, 2))
        t = min(np.abs(length - q), np.abs(length + q))
        if t < ray.tmin:  # Ignore intersection too close
            t = max(np.abs(length - q), np.abs(length + q))
        if t > ray.tmax:  # See if t smaller than max distance
            return None
        ray.tmax = t
        return {"p": ray.p + t * ray.dir, "n": normalize(ray.p + t * ray.dir - self.center), "material": self.material}
def find_hit(scene, ray):
    isect = None
    for primitive in scene:
        t_isect = primitive.intersect(ray)
        if t_isect:
            isect = t_isect
    return isect
def light(p, n, material, scene, emission, l_sample=1):
    if emission and np.abs(p[0] - 50) <= 0.01 and all(np.abs(p[[1, 2]]) <= np.array([20, 20])):  # emission light
        return np.array([1.5, 1.5, 1.5])
    illumination = np.zeros(3)
    for i in range(l_sample):
        light_point = np.array([50., 40*np.random.random() - 20, 40*np.random.random() - 20])
        dir, ray_len = normalize(light_point - p), np.linalg.norm(light_point - p)
        if not find_hit(scene, Ray(p + 0.1 * dir, dir, ray_len - 0.5)):
            illumination += np.abs(np.dot(dir, n)) * np.array([1., 1., 1.])
    return illumination * material["brdf"] / l_sample
def trace_ray(ray, scene, max_iter):
    color, para, emission = np.array([0., 0., 0.]), np.array([1., 1., 1.]),  True
    for iter_num in range(max_iter):
        isect = find_hit(scene, ray)
        if np.linalg.norm(para) <= 0.01 or not isect:       # Break if para too small or not hit anything
            break
        new_dir, p = Sampler[isect["material"]["type"]](-ray.dir, isect["n"])  # Generate new ray
        if not isect["material"]["isdelta"]:
            color += para * light(isect["p"], isect["n"], isect["material"], scene, emission)   # Calculate light
            para *= np.abs(np.dot(new_dir, isect["n"]))
        para *= isect["material"]["brdf"] / p
        emission = isect["material"]["isdelta"]
        ray = Ray(isect["p"] + 0.1 * new_dir, new_dir, tmin=0.11)
    return color
if __name__ == "__main__":
    img, camera_p, sample_num, max_iter = Image.new('RGB', (300, 300)),  np.array([0., 0., -203.]), 3, 4
    my_scene = [Sphere(np.array([-30.,-20.,20.]), {"type": "Mirror","brdf": np.array([1., 1., 1.]),"isdelta": True}, 20), Sphere(np.array([-30., 20., -15.]), {"type": "Glass","brdf": np.array([1., 1., 1.]),"isdelta": True}, 20),
                Sphere(np.array([-40., -10., -30.]), {"type": "Glass", "brdf": np.array([1., 0., 0.]), "isdelta": True}, 10),
                Sphere(np.array([-40., -30., -20.]), {"type": "Glass", "brdf": np.array([1., 1., 0.]), "isdelta": True}, 10),
                Plane(50, np.array([50, 0, 0]), np.array([-1., 0., 0.]), [1, 2]), Plane(50, np.array([-50, 0, 0]), np.array([1., 0., 0.]), [1, 2]), Plane(50, np.array([0, 0, 50]), np.array([0., 0., -1.]), [0, 1]),
                Plane(50, np.array([0, 50, 0]), np.array([0., -1., 0.]), [0, 2], Red), Plane(50, np.array([0, -50, 0]), np.array([0., 1., 0.]), [0, 2], Blue)]
    for h in range(img.height):
        print(h)
        for w in range(img.width):
            color = np.zeros(3)
            for sample in range(sample_num):
                canvas_point = np.array([1. - 2. * (h+np.random.random()) / img.height, 2. * (w+np.random.random()) / img.width - 1., -200])
                color += trace_ray(Ray(camera_p, canvas_point - camera_p), my_scene, max_iter)
            img.putpixel((w, h), tuple(int(255 * i / sample_num) for i in color))
    img.show()