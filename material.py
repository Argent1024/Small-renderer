import numpy as np


def normalize(x):
    return x / np.linalg.norm(x)


def reflect(in_dir, normal):
    return 2. * np.dot(normal, in_dir) * normal - in_dir


class Sampler:
    @staticmethod
    def glass_sampler(in_dir, normal, eta=1/1.45):
        cos = np.dot(in_dir, normal)
        if cos < 0:         # Ray from inside
            eta = 1 / eta
            cos, normal = np.abs(cos), -normal
        t = 1 - np.power(eta, 2) * (1 - np.power(cos, 2))

        if t < 0:
            # total reflection
            return reflect(in_dir, normal), 1

        r0 = np.power(eta - 1, 2) / np.power(eta + 1, 2)
        p = r0 + (1. - r0) * np.power(1. - cos, 5)    # Schlick's approximation
        new_dir = None
        if np.random.random() < p:
            # reflection
            new_dir = reflect(in_dir, normal)
            p = 1.
        else:
            # refraction
            t = 1 - np.power(eta, 2) * (1 - np.power(cos, 2))
            new_dir = -eta * in_dir + (eta * cos - np.sqrt(t)) * normal
            p = 1.
        return new_dir, p

    @staticmethod
    def mirror_sampler(in_dir, normal):
        return reflect(in_dir, normal), 1.

    @staticmethod
    def cos_weight_hemisphere(in_dir, n):
        theta = 2 * np.pi * np.random.random()
        cos_p = np.random.random()
        v = normalize(np.array([0, 0, 1]) - n[2] * n) if (n[2] != 1 and n[2] != -1) else np.array([1, 0, 0])
        sin_p = np.sqrt(1 - cos_p * cos_p)
        T = np.transpose(np.array([v, np.cross(v, n), n]))
        new_dir = np.matmul(T, np.array([np.sin(theta)*sin_p, np.cos(theta)*sin_p, cos_p]))
        return new_dir, cos_p / np.pi

    @staticmethod
    def get_sampler(material_type):
        d = {
            "Diffuse": Sampler.cos_weight_hemisphere,
            "Glass": Sampler.glass_sampler,
            "Mirror": Sampler.mirror_sampler
        }
        return d[material_type]


class Material:
    def __init__(self, brdf, delta, mat_type):
        self._brdf = brdf
        self._delta = delta
        self.mat_type = mat_type

    def brdf(self):
        return self._brdf

    def sample(self, dir, normal):
        sampler = Sampler.get_sampler(self.mat_type)
        return sampler(dir, normal)

    def is_delta(self):
        return self._delta


Mirror = Material(np.array([1., 1., 1.]), True, "Mirror")

DefaultMat = Material(np.array([1., 1., 1.]) / np.pi, False, "Diffuse")
RedWall = Material(np.array([1.0, 0.75, 0.75]) / np.pi, False, "Diffuse")
BlueWall = Material(np.array([0.75, 0.75, 1.0]) / np.pi, False, "Diffuse")

WhiteGlass = Material(np.array([1., 1., 1.]), True, "Glass")
RedGlass = Material(np.array([1., 0., 0.]), True, "Glass")
YellowGlass = Material(np.array([1., 1., 0.]), True, "Glass")
