# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import math
import mkl_fft

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile


@PIPELINES.register_module()
class LoadRadarsFromFile(object):
    """Load Radars From File.

    Load sunrgbd and scannet points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        radar_dim (int): The dimension of the input radar.
            Defaults to [512, 256, 16].
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(self,
                 coord_type,
                 unwrap_tx=False,
                 adc2rt=False,
                 convert_complex=True,
                 modality_type='RT',
                 numSamplePerChirp=512,
                 numChirps=256,
                 numReducedDoppler=16,
                 numChirpsPerLoop=16,
                 radar_dim=[512, 256, 16],
                 shift_height=False,
                 file_client_args=dict(backend='disk')):
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']
        assert modality_type in ['ADC', 'RT', 'RD', 'RA', 'PCD']
        
        self.coord_type = coord_type
        self.unwrap_tx = unwrap_tx
        self.convert_complex = convert_complex
        self.modality_type = modality_type
        self.numSamplePerChirp = numSamplePerChirp
        self.numChirps = numChirps
        self.numReducedDoppler = numReducedDoppler
        self.numChirpsPerLoop = numChirpsPerLoop
        self.radar_dim = radar_dim
        self.shift_height = shift_height
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.adc2rt = adc2rt

        # Build hamming window table to reduce side lobs
        hanningWindowRange = (0.54 - 0.46*np.cos(((2*math.pi*np.arange(self.numSamplePerChirp ))/(self.numSamplePerChirp -1))))
        hanningWindowDoppler = (0.54 - 0.46*np.cos(((2*math.pi*np.arange(self.numChirps ))/(self.numChirps -1))))
        self.range_fft_coef = np.expand_dims(np.repeat(np.expand_dims(hanningWindowRange,1), repeats=self.numChirps, axis=1),2)
        self.doppler_fft_coef = np.expand_dims(np.repeat(np.expand_dims(hanningWindowDoppler, 1).transpose(), repeats=self.numSamplePerChirp, axis=0),2)

        # build coefficients for phase shift
        doppler_bin = 0
        dividend_constant_arr = np.arange(0, self.numReducedDoppler*self.numChirpsPerLoop, self.numReducedDoppler)
        DopplerBinSeq = np.remainder(doppler_bin + dividend_constant_arr,  self.numChirps)
        DopplerBinSeq = np.concatenate([[DopplerBinSeq[0]],DopplerBinSeq[5:]])[None, :]
        SlowTimeBinSeq = np.array(np.arange(start=0, stop= self.numChirps, step=1))[:, None]
        self.phase_shift = np.exp(-1j*2*math.pi*DopplerBinSeq*SlowTimeBinSeq/self.numChirps)

    def _load_radars_adc(self, radars_adc_filename):
        """Private function to load radars data, which has been processed by range fft.

        Args:
            radars_adc_filename (str): Filename of radars data.

        Returns:
            np.ndarray: An array containing radars data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(radars_adc_filename)
            radars_adc = np.frombuffer(pts_bytes, dtype=np.complex128)
        except ConnectionError:
            mmcv.check_file_exist(radars_adc_filename)
            if radars_adc_filename.endswith('.npy'):
                radars_adc = np.load(radars_adc_filename)
            else:
                radars_adc = np.fromfile(radars_adc_filename, dtype=np.complex128)

        radars_adc = radars_adc.reshape(self.radar_dim[0], self.radar_dim[1], -1)
        if self.adc2rt:
            # mainly for testing if adc data is correct
            radars_adc = np.multiply(radars_adc,self.range_fft_coef)

            # for debug only
            # radars_range = mkl_fft.fft(np.multiply(radars_adc,self.range_fft_coef),self.numSamplePerChirp,axis=0)
            # radars_win = np.multiply(radars_adc,self.range_fft_coef)
            # radars_adc = np.concatenate((radars_win, radars_range), axis=2)
            # self.radar_dim[-1] = 32

        if self.convert_complex:
            adc_angle = np.angle(radars_adc)
            adc_mag = np.abs(radars_adc)
            radars_adc = np.stack([adc_angle, adc_mag], axis=-1).astype(dtype=np.float32)
        else:
            adc_real = np.real(radars_adc)
            adc_imag = np.imag(radars_adc)
            radars_adc = np.stack([adc_real, adc_imag], axis=-1).astype(dtype=np.float32)
        
        radars_adc = radars_adc.reshape(self.radar_dim[0], self.radar_dim[1], -1)
        assert radars_adc.shape[-1] == self.radar_dim[-1]*2, "Shape not match on last dimension." 

        return radars_adc

    def _load_radars_rt(self, radars_rt_filename):
        """Private function to load radars data, which has been processed by range fft.

        Args:
            radars_rt_filename (str): Filename of radars data.

        Returns:
            np.ndarray: An array containing radars data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(radars_rt_filename)
            radars_rt = np.frombuffer(pts_bytes, dtype=np.complex128)
        except ConnectionError:
            mmcv.check_file_exist(radars_rt_filename)
            if radars_rt_filename.endswith('.npy'):
                radars_rt = np.load(radars_rt_filename)
            else:
                radars_rt = np.fromfile(radars_rt_filename, dtype=np.complex128)

        radars_rt = radars_rt.reshape(self.radar_dim[0], self.radar_dim[1], -1)

        if self.unwrap_tx:
            win_rt = np.multiply(radars_rt, self.doppler_fft_coef)
            radars_rt = np.multiply(win_rt[:, :, None, :], self.phase_shift[None, :, :, None])
            radars_rt = radars_rt.reshape(self.radar_dim[0], self.radar_dim[1], -1)
            self.radar_dim[-1] = 192

        if self.convert_complex:
            rt_angle = np.angle(radars_rt)
            rt_mag = np.abs(radars_rt)
            radars_rt = np.stack([rt_mag, rt_angle], axis=-1).astype(dtype=np.float32)
        else:
            rt_real = np.real(radars_rt)
            rt_imag = np.imag(radars_rt)
            radars_rt = np.stack([rt_real, rt_imag], axis=-1).astype(dtype=np.float32)
        
        radars_rt = radars_rt.reshape(self.radar_dim[0], self.radar_dim[1], -1)
        assert radars_rt.shape[-1] == self.radar_dim[-1]*2, "Shape not match on last dimension." 

        return radars_rt
    
    def _load_radars_rd(self, radars_rd_filename):
        """Private function to load radars data, which has been processed by range fft.

        Args:
            radars_rd_filename (str): Filename of radars data.

        Returns:
            np.ndarray: An array containing radars data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(radars_rd_filename)
            radars_rd = np.frombuffer(pts_bytes, dtype=np.complex128)
        except ConnectionError:
            mmcv.check_file_exist(radars_rd_filename)
            if radars_rd_filename.endswith('.npy'):
                radars_rd = np.load(radars_rd_filename)
            else:
                radars_rd = np.fromfile(radars_rd_filename, dtype=np.complex128)

        radars_rd = radars_rd.reshape(self.radar_dim[0], self.radar_dim[1], -1) 

        if self.convert_complex:
            rt_angle = np.angle(radars_rd)
            rt_mag = np.abs(radars_rd)
            radars_rd = np.stack([rt_mag, rt_angle], axis=-1).astype(dtype=np.float32)
        else:
            rt_real = np.real(radars_rd)
            rt_imag = np.imag(radars_rd)
            radars_rd = np.stack([rt_real, rt_imag], axis=-1).astype(dtype=np.float32)
        
        radars_rd = radars_rd.reshape(self.radar_dim[0], self.radar_dim[1], -1)
        assert radars_rd.shape[-1] == self.radar_dim[-1]*2, "Shape not match on last dimension." 

        return radars_rd
    
    def _load_radars_ra(self, radars_ra_filename):
        """Private function to load Range-Azimuth map of radar.

        Args:
            radars_ra_filename (str): Filename of radars data.

        Returns:
            np.ndarray: An array containing radars data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(radars_ra_filename)
            radars_ra = np.frombuffer(pts_bytes, dtype=np.float64)
        except ConnectionError:
            mmcv.check_file_exist(radars_ra_filename)
            if radars_ra_filename.endswith('.npy'):
                radars_ra = np.load(radars_ra_filename)
            else:
                radars_ra = np.fromfile(radars_ra_filename, dtype=np.float64)
        
        #  shape for RADIal is [512, 751]
        radars_ra = radars_ra.reshape(self.radar_dim[0], self.radar_dim[1], -1).astype(dtype=np.float32) 
        assert radars_ra.shape[-1] == self.radar_dim[-1], "Shape not match on last dimension."

        return radars_ra
    
    def _load_radars_pcd(self, radars_pcd_filename):
        """Private function to load radar pcd.

        Args:
            radars_pcd_filename (str): Filename of radars data.

        Returns:
            np.ndarray: An array containing radars data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(radars_pcd_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(radars_pcd_filename)
            if radars_pcd_filename.endswith('.npy'):
                points = np.load(radars_pcd_filename)
            else:
                points = np.fromfile(radars_pcd_filename, dtype=np.float32)

        return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        radar_filename = results['radar_filename']
        if self.modality_type == 'ADC':
            radars_adc = self._load_radars_adc(radar_filename)
            results['radars_adc'] = [radars_adc]
        elif self.modality_type == 'RT':
            radars_rt = self._load_radars_rt(radar_filename)  # [R, D, 2*C]
            results['radars_rt'] = [radars_rt]
        elif self.modality_type == 'RD':
            radars_rd = self._load_radars_rd(radar_filename)  # [R, D, 2*C]
            results['radars_rt'] = [radars_rd]  # TODO: unify the name
        elif self.modality_type == 'RA':
            radars_ra = self._load_radars_ra(radar_filename)  # [R, D, C]
            results['radars_ra'] = [radars_ra]
        elif self.modality_type == 'PCD':
            pc = self._load_radars_pcd(radar_filename)
            pc = pc.reshape(self.radar_dim)  # [N, C]
            r = pc[:,0]
            az = pc[:,2]
            el = pc[:,3]
            x = r*np.cos(az)*np.cos(el)
            y = r*np.sin(az)*np.cos(el)
            z = r*np.sin(el)
            points = np.stack([r, az, z], axis=-1)
            attribute_dims = None

            if self.shift_height:
                floor_height = np.percentile(points[:, 2], 0.99)
                height = points[:, 2] - floor_height
                points = np.concatenate(
                    [points[:, :3],
                    np.expand_dims(height, 1), points[:, 3:]], 1)
                attribute_dims = dict(height=3)

            points_class = get_points_type(self.coord_type)
            points = points_class(
                points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
            results['points'] = points
        else:
            print(self.modality_type)
            raise NotImplementedError

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'file_client_args={self.file_client_args}, '
        repr_str += f'radar_dim={self.radar_dim}, '
        return repr_str


@PIPELINES.register_module()
class LoadPolarPointsFromFile(object):
    """Load Points From File.

    Load sunrgbd and scannet points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int]): Which dimensions of the points to be used.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool): Whether to use shifted height. Defaults to False.
        use_color (bool): Whether to use color features. Defaults to False.
        use_polar (bool): Whether to replace first 2 dim with range and azimuth.
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(self,
                 coord_type,
                 load_dim=6,
                 use_dim=[0, 1, 2],
                 shift_height=False,
                 use_color=False,
                 use_polar=True,
                 file_client_args=dict(backend='disk')):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.use_polar = use_polar
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_filename = results['pts_filename']
        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        if self.use_polar:
            r = np.linalg.norm(points[:, :2], axis=-1)
            a = np.arctan2(points[:, 1], points[:, 0])
            points[:, 0] = r
            points[:, 1] = a
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'file_client_args={self.file_client_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        return repr_str


@PIPELINES.register_module()
class AddRadarPointsFromFile(object):
    """Add radar points to LiDAR.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        radar_dim (int): The dimension of the input radar.
            Defaults to [512, 256, 16].
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(self,
                 coord_type,
                 unwrap_tx=False,
                 convert_complex=True,
                 modality_type='PCD',
                 numReducedDoppler=16,
                 numChirpsPerLoop=16,
                 radar_dim=[512, 256, 16],
                 shift_height=False,
                 file_client_args=dict(backend='disk')):
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']
        assert modality_type in ['PCD']
        
        self.coord_type = coord_type
        self.unwrap_tx = unwrap_tx
        self.convert_complex = convert_complex
        self.modality_type = modality_type
        self.numReducedDoppler = numReducedDoppler
        self.numChirpsPerLoop = numChirpsPerLoop
        self.radar_dim = radar_dim
        self.shift_height = shift_height
        self.file_client_args = file_client_args.copy()
        self.file_client = None
    
    def _load_radars_pcd(self, radars_pcd_filename):
        """Private function to load radar pcd.

        Args:
            radars_pcd_filename (str): Filename of radars data.

        Returns:
            np.ndarray: An array containing radars data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(radars_pcd_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(radars_pcd_filename)
            if radars_pcd_filename.endswith('.npy'):
                points = np.load(radars_pcd_filename)
            else:
                points = np.fromfile(radars_pcd_filename, dtype=np.float32)

        return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        radar_filename = results['radar_filename']
        if self.modality_type == 'PCD':
            pc = self._load_radars_pcd(radar_filename)
            pc = pc.reshape(self.radar_dim)  # [N, C]
            r = pc[:,0]
            az = pc[:,2]
            el = pc[:,3]
            x = r*np.cos(az)*np.cos(el)
            y = r*np.sin(az)*np.cos(el)
            z = r*np.sin(el) + 0.8 - 0.42 # 0.8 is the height of radar, 0.42 is the height of LiDAR
            points = np.stack([r, az, z], axis=-1)
            attribute_dims = None

            lidar_dim = results['points'].points_dim
            extra_dim = lidar_dim - points.shape[-1]
            if extra_dim > 0:
                extra_attr = np.zeros((points.shape[0], extra_dim), dtype=np.float32)
                points = np.concatenate([points, extra_attr], axis=-1)

            if self.shift_height:
                floor_height = np.percentile(points[:, 2], 0.99)
                height = points[:, 2] - floor_height
                points = np.concatenate(
                    [points[:, :3],
                    np.expand_dims(height, 1), points[:, 3:]], 1)
                attribute_dims = dict(height=3)

            # cat lidar and radar points
            points = np.concatenate([np.array(results['points'].tensor), points], axis=0)
            points_class = get_points_type(self.coord_type)
            points = points_class(
                points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
            
            results['points'] = points
        else:
            raise NotImplementedError

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'file_client_args={self.file_client_args}, '
        repr_str += f'radar_dim={self.radar_dim}, '
        return repr_str

@PIPELINES.register_module()
class LoadKRadarsFromFile(object):
    """Load K-Radar Data From File.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        radar_dim (int): The dimension of the input radar.
            Defaults to [512, 256, 16].
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(self,
                 coord_type,
                 cube_range=[0, -80, -30, 100, 80, 30],
                 modality_type='PCD',
                 convert2polar=True,
                 shift_height=False,
                 file_client_args=dict(backend='disk')):
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']
        assert modality_type in ['RA', 'PCD']
        
        self.coord_type = coord_type
        self.cube_range = cube_range
        self.modality_type = modality_type
        self.convert2polar = convert2polar
        self.shift_height = shift_height
        self.file_client_args = file_client_args.copy()
        self.file_client = None
    
    def _load_radars_ra(self, radars_ra_filename):
        """Private function to load Range-Azimuth map of radar.

        Args:
            radars_ra_filename (str): Filename of radars data.

        Returns:
            np.ndarray: An array containing radars data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        radars_ra = np.load(radars_ra_filename).astype(dtype=np.float32)

        return radars_ra
    
    def _load_radars_pcd(self, radars_pcd_filename):
        """Private function to load radar pcd.

        Args:
            radars_pcd_filename (str): Filename of radars data.

        Returns:
            np.ndarray: An array containing radars data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        points = np.load(radars_pcd_filename).astype(dtype=np.float32)

        return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        radar_filename = results['radar_filename']
        if self.modality_type == 'RA':
            radars_ra = self._load_radars_ra(radar_filename)  # [R, D, C]
            results['radars_ra'] = [radars_ra]
            
        elif self.modality_type == 'PCD':
            pc = self._load_radars_pcd(radar_filename)  # [N, C]
            if self.convert2polar:
                r = np.sqrt(np.sum(pc[:, :3]**2, axis=1))
                az = np.arctan2(pc[:, 1], pc[:, 0])
                z = pc[:, 2]
                pw = pc[:, 3]
                points = np.stack([r, az, z, pw], axis=-1)
            else:
                points = pc
            
            attribute_dims = None

            if self.shift_height:
                floor_height = np.percentile(points[:, 2], 0.99)
                height = points[:, 2] - floor_height
                points = np.concatenate(
                    [points[:, :3],
                    np.expand_dims(height, 1), points[:, 3:]], 1)
                attribute_dims = dict(height=3)

            points_class = get_points_type(self.coord_type)
            points = points_class(
                points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
            results['points'] = points
        else:
            print(self.modality_type)
            raise NotImplementedError

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'file_client_args={self.file_client_args}, '
        repr_str += f'radar_dim={self.radar_dim}, '
        return repr_str