"""Canonical spherical groups, native inputs and explicit dense voxel witnesses."""
from dataclasses import replace
import math
import unittest

import numpy as np

from XTA.config import TiltedViewGroup
from XTA import geometry, spherical_geometry as sphere


def views(shape=(7,9,11), size=8, minimum=.7, targets=('transverse',), groups=()):
    return [v for v in geometry.get_view_infos(*shape,cartesian_views=(),
        spherical_views=targets,spherical_min_radius=minimum,spherical_patch_size=size,
        tilt_groups=groups) if v.family=='spherical']


class SphericalGeometryTests(unittest.TestCase):
    def test_upright_aliases_have_identical_frames_and_stable_order(self):
        source=np.random.default_rng(6).integers(0,256,(7,9,11),dtype=np.uint8)
        baseline=views()
        for tokens in (('sagittal',),('coronal',),('coronal','transverse','sagittal')):
            current=views(targets=tokens)
            self.assertEqual([v.name for v in current],[v.name for v in baseline])
            for first,second in zip(baseline,current):
                np.testing.assert_array_equal(sphere.render_shell_frame(source,first,0),
                                               sphere.render_shell_frame(source,second,0))
        self.assertEqual(views(targets=('coronal','transverse','sagittal'))[0].spherical_request_tokens,
                         ('transverse','sagittal','coronal'))

    def test_signed_direction_groups_dedupe_bases_but_remain_distinct(self):
        groups=(TiltedViewGroup(('transverse','sagittal','coronal'),(30.,),('vertical','horizontal')),)
        one=views(size=16,targets=('tilted_transverse',),groups=groups)
        all_views=views(size=16,targets=('tilted_coronal','tilted_sagittal','tilted_transverse'),groups=groups)
        self.assertEqual([v.name for v in one],[v.name for v in all_views])
        self.assertEqual(len(all_views),24)
        self.assertEqual(len({v.spherical_group for v in all_views}),4)
        for view in all_views:
            matrix=np.asarray(view.spherical_rotation_xyz).reshape(3,3)
            np.testing.assert_allclose(matrix.T@matrix,np.eye(3),atol=2e-15)
            self.assertAlmostEqual(np.linalg.det(matrix),1.)
            self.assertEqual(len(view.spherical_request_tokens),3)

    def test_radius_defaults_are_independent_and_sphere_uses_shortest_axis(self):
        combined=geometry.get_view_infos(11,15,17,cartesian_views=(),radial_views=('transverse',),
            radial_min_radius=3.,radial_patch_size=8,spherical_views=('transverse',),spherical_patch_size=8)
        view=next(v for v in combined if v.family=='spherical')
        self.assertEqual(view.spherical_min_radius,8/(4*math.pi))
        self.assertEqual(view.spherical_max_radius,5.)
        self.assertEqual(view.spherical_radii[0],view.spherical_min_radius)
        self.assertEqual(view.spherical_radii[-1],5.)
        self.assertTrue(np.all(np.diff(view.spherical_radii)<=1.))

    def test_patches_cover_fixed_face_lattice_and_small_faces_are_padded(self):
        for size in (6,16):
            selected=views(size=size)
            n=selected[0].spherical_face_intervals
            for face in range(6):
                coverage=np.zeros((n+1,n+1),bool)
                for view in selected:
                    if view.spherical_face!=face: continue
                    rows=np.arange(size)+view.spherical_v_origin
                    cols=np.arange(size)+view.spherical_u_origin
                    coverage[np.ix_(rows[(rows>=0)&(rows<=n)],cols[(cols>=0)&(cols<=n)])]=True
                self.assertTrue(coverage.all())
            first=selected[0]
            _,valid=sphere.face_directions(first)
            rendered=sphere.render_shell_frame(np.full((7,9,11),255,np.uint8),first,0)
            self.assertTrue(np.all(rendered[~valid]==0))

    def test_every_annular_voxel_has_an_actual_positive_input_tap(self):
        for shape in ((7,9,11),(6,8,10),(3,5,7)):
            groups=(TiltedViewGroup(('transverse',),(31.,),('vertical','horizontal')),)
            selected=views(shape=shape,size=6,minimum=.3,targets=('transverse','tilted_transverse'),groups=groups)
            zz,yy,xx=np.indices(shape,dtype=np.float64)
            radius=np.sqrt((zz-(shape[0]-1)/2)**2+(yy-(shape[1]-1)/2)**2+(xx-(shape[2]-1)/2)**2)
            domain=(radius>=.3)&(radius<=(min(shape)-1)/2)
            for group in {v.spherical_group for v in selected}:
                witnessed=np.zeros(shape,bool)
                for view in selected:
                    if view.spherical_group!=group: continue
                    for index in range(view.num_slices):
                        t,y,x,valid=sphere.shell_coordinates(view,index)
                        coords=(t,y,x)
                        low=tuple(np.floor(c).astype(np.int64) for c in coords)
                        frac=tuple(c-l for c,l in zip(coords,low))
                        for it in (0,1):
                            for iy in (0,1):
                                for ix in (0,1):
                                    at=tuple(l+d for l,d in zip(low,(it,iy,ix)))
                                    weight=(frac[0] if it else 1-frac[0])*(frac[1] if iy else 1-frac[1])*(frac[2] if ix else 1-frac[2])
                                    active=valid&(weight>0)
                                    for a,n in zip(at,shape): active&=(a>=0)&(a<n)
                                    witnessed[tuple(a[active] for a in at)]=True
                self.assertFalse(np.any(domain&~witnessed),(shape,group))

    def test_radius_channels_clamp_inside_same_face_patch(self):
        view=views()[0]
        for index in (-5,-1,view.num_slices,view.num_slices+7):
            mapped=geometry.channel_view_slice_source(view,index)
            self.assertEqual(mapped,(max(0,min(view.num_slices-1,index)),False))

    def test_categorical_rendering_remains_binary(self):
        source=np.random.default_rng(2).integers(0,2,(7,9,11),dtype=np.uint8)*255
        for view in views(size=6):
            result=geometry.get_categorical_view_frame_by_index(source,view,0)
            self.assertTrue(np.all((result==0)|(result==1)))


if __name__=='__main__':
    unittest.main()
