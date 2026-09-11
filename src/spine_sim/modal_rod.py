"""Nonlinear Ritz reduction of the existing variable-section guided rod.

Two bending shapes per transverse direction span the discrete linear responses
to a tip force and a tip moment. Geometry/energy still use every declared rod
segment; only the solved coordinates are reduced. The basis follows exposed
length, including its contribution to the compression derivative.
"""
from __future__ import annotations

import numpy as np
from functools import lru_cache

from .guided_rod import GuidedRod, RodEvaluation


class ModalGuidedRod(GuidedRod):
    reduction = "force_moment_ritz"

    @property
    def dimension(self):
        return 5

    @lru_cache(maxsize=32)
    def bending_basis(self, compression_fraction):
        p = self.parameters
        length = p.free_length_m * (1.-compression_fraction)
        boundaries = length*self.segment_fractions
        h = np.diff(boundaries)
        inertia = p.integrated_second_moment_m5(length-boundaries[1:], length-boundaries[:-1])
        weights = p.young_modulus_Pa*inertia/h**2
        # D.T diag(weights) D is the zero-curvature angular Hessian.
        # Its inverse on the tip force/moment columns is obtained by sums.
        force_shape = np.cumsum((length-.5*(boundaries[:-1]+boundaries[1:]))/weights)
        moment_shape = np.cumsum(1./weights)
        shapes = np.column_stack((force_shape, moment_shape))
        basis, triangular = np.linalg.qr(shapes, mode="reduced")
        return basis*np.sign(np.diag(triangular))

    def expand_state(self, state):
        x = np.asarray(state, dtype=float)
        angular = self.bending_basis(x[0]) @ x[1:].reshape(2, 2)
        return np.concatenate(([x[0]], angular.ravel()))

    def _kinematics(self, state):
        return GuidedRod._kinematics(self, self.expand_state(state))

    def evaluate(self, state, derivatives=True):
        x = np.asarray(state, dtype=float)
        p = self.parameters
        full = self.expand_state(x)
        center, rotation, bending, centerline, tangents = GuidedRod._kinematics(self, full)
        compression = p.free_length_m*x[0]
        stiffness = p.spring_stiffness_N_per_m if p.mount_type == "spring" else 0.
        spring = .5*stiffness*compression**2
        gradient = jacobian = rotation_jacobian = None
        if derivatives:
            steps = self.derivative_step*np.maximum(1., np.abs(x[1:]))
            trials = np.tile(x[1:], (8, 1))
            indices = np.arange(4)
            trials[indices, indices] += steps
            trials[4+indices, indices] -= steps
            angular = np.einsum("jk,nkd->njd", self.bending_basis(x[0]), trials.reshape(8, 2, 2)).reshape(8, -1)
            centers, rotations, energies = GuidedRod._angular_kinematics_batch(self, full, angular)
            jacobian = np.empty((3, 5))
            rotation_jacobian = np.empty((3, 5))
            gradient = np.empty(5)
            jacobian[:, 1:] = ((centers[:4]-centers[4:])/(2*steps[:, None])).T
            spin = ((rotations[:4]-rotations[4:])/(2*steps[:, None, None])) @ rotation.T
            rotation_jacobian[:, 1:] = np.stack((spin[:, 2, 1]-spin[:, 1, 2],
                                                 spin[:, 0, 2]-spin[:, 2, 0],
                                                 spin[:, 1, 0]-spin[:, 0, 1]))/2
            gradient[1:] = (energies[:4]-energies[4:])/(2*steps)
            step = min(self.derivative_step, .2*(1.-x[0]))
            plus, minus = x.copy(), x.copy()
            plus[0] += step
            minus[0] -= step
            cp, rp, ep, _, _ = self._kinematics(plus)
            cm, rm, em, _, _ = self._kinematics(minus)
            jacobian[:, 0] = (cp-cm)/(2*step)
            spin0 = ((rp-rm)/(2*step)) @ rotation.T
            rotation_jacobian[:, 0] = np.array((spin0[2, 1]-spin0[1, 2],
                                                spin0[0, 2]-spin0[2, 0],
                                                spin0[1, 0]-spin0[0, 1]))/2
            gradient[0] = (ep-em)/(2*step)+stiffness*p.free_length_m*compression
        return RodEvaluation(center, rotation, spring+bending, spring, bending,
                             compression, p.free_length_m-compression, jacobian,
                             rotation_jacobian, gradient, centerline, tangents)
