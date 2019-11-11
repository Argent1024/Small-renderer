import numpy as np
from PIL import Image
normalize = lambda x: x / np.linalg.norm(x)
reflect = lambda in_dir, normal: 2. * normal - in_dir
refract = lambda in_dir, normal: 0
def cosWeightHemiSPhere(in_dir, n):
    theta, cos_p = 2 * np.pi * np.random.random(), np.random.random()
    v = normalize(np.array([0, 0, 1]) - n[2] * n) if (n[2] != 1 and n[2] != -1) else np.array([1, 0, 0])
    sin_p, T = np.sqrt(1 - cos_p * cos_p), np.transpose(np.array([v,np.cross(v, n),n]))
    return np.matmul(T, np.array([np.sin(theta)*sin_p, np.cos(theta)*sin_p, cos_p])), cos_p
def glassSampler(in_dir, n, eta=0.8):
    return reflect(in_dir, n)[0], 0.8 if np.random.random() < eta else refract(in_dir, n)

Sampler = {"Diffuse": cosWeightHemiSPhere, "Glass": glassSampler, "Mirror": lambda i,n: (reflect(1, n), 1)}
Mirror = {"type":"Mirror", "brdf":np.array([1.,1.,1.]), "isdelta": True}
Glass = {"type":"Glass", "brdf":np.array([1.,1.,1.]), "isdelta": True}
Diffuse = {"type": "Diffuse", "brdf": np.array([0.6, 0.6, 0.6]), "isdelta": False}
Red = {"type": "Diffuse", "brdf": np.array([0.75, 0.6, 0.6]), "isdelta": False}
Blue = {"type": "Diffuse", "brdf": np.array([0.6, 0.6, 0.75]), "isdelta": False}

class Ray:
    def __init__(self, p, d, t=np.finfo('f').max):
        self.dir, self.p, self.t = normalize(d), p, t

class Plane: # A hard code plane, center can only be (x,0,0), (0,y,0), (0,0,z)
    def __init__(self, length, center, normal, axis, material):  # which axises are not zero
        self.length, self.center, self.normal, self.zeroaxis, self.index, self.material \
            = length, center, normal, axis, {0, 1, 2}.difference(set(axis)).pop(), material
    def intersect(self, ray):  # intersect of ray and box, return t/f, isect point, normal
        if ray.dir[self.index] != 0:
            length = (self.center[self.index] - ray.p[self.index]) / ray.dir[self.index]
            p = ray.p + length * ray.dir
            if 0 < length < ray.t and all(np.abs(p[self.zeroaxis]) <= np.array([self.length, self.length])):
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
        if t > ray.t:
            return None
        ray.t = t
        return {"p": ray.p + t * ray.dir, "n": normalize(ray.p + t * ray.dir - self.center), "material": self.material}

def find_hit(scene, ray):
    isect = None
    for primitive in scene:
        t_isect = primitive.intersect(ray)
        if t_isect:
            isect = t_isect
    return isect

def light(p, n, material, scene, emission):
    if emission and np.abs(p[0] - 50) <= 0.01 and all(np.abs(p[[1, 2]]) <= np.array([20, 20])):  # emission light
        return np.array([1., 1., 1.])
    dir = normalize(np.array([50., 0, 0]) - p)
    ray_len = np.linalg.norm(np.array([50., 0, 0]) - p)
    ray = Ray(p + 0.1 * dir, dir, ray_len - 0.5)
    return np.zeros(3) if find_hit(scene, ray) else np.abs(np.dot(dir, n)) * material["brdf"]

def trace_ray(ray, scene, max_iter):
    color, para, emission = np.array([0., 0., 0.]), np.array([1., 1., 1.]),  True
    for iter_num in range(max_iter):
        isect = find_hit(scene, ray)
        if not isect:
            break
        if not isect["material"]["isdelta"]:
            color += para * light(isect["p"], isect["n"], isect["material"], scene, emission)   # Calculate light
        new_dir, p = Sampler[isect["material"]["type"]](ray.dir, isect["n"])                 # Generate new ray
        para *= np.abs(np.dot(-new_dir, isect["n"])) / p * isect["material"]["brdf"]
        emission = isect["material"]["isdelta"]
        ray = Ray(isect["p"] + 0.1 * new_dir, new_dir)
    return color

if __name__ == "__main__":
    img, camera_p = Image.new('RGB', (300, 300)),  np.array([0., 0., -203.])
    my_scene = [Sphere(np.array([-35., 15., 30.]), Mirror, 15), Sphere(np.array([-35., -18., -20.]), Diffuse, 15),
                Plane(50, np.array([50, 0, 0]), np.array([-1., 0., 0.]), [1, 2], Diffuse),  # Celling
                Plane(50, np.array([0, 50, 0]), np.array([0., -1., 0.]), [0, 2], Red),
                Plane(50, np.array([0, -50, 0]), np.array([0., 1., 0.]), [0, 2], Blue),
                Plane(50, np.array([-50, 0, 0]), np.array([1., 0., 0.]), [1, 2], Diffuse),
                Plane(50, np.array([0, 0, 50]), np.array([0., 0., -1.]), [0, 1], Diffuse)]
    sample_num, max_iter = 1, 4
    for h in range(img.height):
        for w in range(img.width):
            color = np.zeros(3)
            for sample in range(sample_num):
                canvas_point = np.array([1. - 2. * h / img.height, 2. * w / img.width - 1., -200])
                color += trace_ray(Ray(camera_p, canvas_point - camera_p), my_scene, max_iter)
            img.putpixel((w, h), tuple(int(255 * i / sample_num) for i in color))
    img.show()
    img.save("D://test.png")