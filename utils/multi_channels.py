# SPDX-FileCopyrightText: Copyright (c) 2026 Mahdi Abdollahpour <mahdi.abdollahpour@unibo.it>

"""Per-sample random TDL and CDL channels (3GPP TR 38.901, Sec. 7.7).

MultiTDLChannel and MultiCDLChannel draw, for every (batch sample, user) link,
a channel profile uniformly from a set of TDL or CDL profiles (A..E unless a
subset is given), a delay spread uniformly from [delay_spread_min,
delay_spread_max], and a UT speed uniformly from [min_speed, max_speed].
They are drop-in replacements for NTDLChannel / NCDLChannel:
``y, h_freq = layer((x, no))``.

The profiles have different numbers of paths, so their constants are padded
with zero-power paths to the longest profile of the chosen set and picked per
link with a gather. A zero-power path contributes nothing, so each link
follows the Sionna model of its profile exactly. A single profile needs no
padding and no gather and costs exactly that profile.

The profile constants are read from the Sionna fork's TDL / CDL objects and
the CDL links use its ChannelCoefficientsGenerator, so the statistics are the
fork's. All shapes depend only on the profile set and the number of users,
and the batch size may be dynamic, so the layers run in graph mode and
under XLA.

Depends only on TensorFlow and Sionna; the package __init__ puts the Sionna
fork on sys.path.
"""

import numpy as np
import tensorflow as tf
from tensorflow import cos, sin
from tensorflow.keras.layers import Layer

import sionna
from sionna import PI, SPEED_OF_LIGHT
from sionna.channel import ApplyOFDMChannel
from sionna.channel.tr38901 import CDL, TDL, Rays, Topology
from sionna.channel.utils import exp_corr_mat, subcarrier_frequencies
from sionna.utils import expand_to_rank, matrix_sqrt


PROFILES = ("A", "B", "C", "D", "E")


###########################################################################
# Helpers shared by both families
###########################################################################

def _profile_set(models):
    """Profiles to draw from: all of A..E if ``models`` is None, else the
    given letters without repeats (first occurrence kept)."""
    if models is None:
        return list(PROFILES)
    if isinstance(models, str):
        models = [models]
    chosen = list(dict.fromkeys(str(m).upper() for m in models))
    if not chosen or any(m not in PROFILES for m in chosen):
        raise ValueError(f"Channel profiles must be a non-empty subset of "
                         f"{PROFILES}, got {models}.")
    return chosen


def _pad_paths(x, num_paths, value=None):
    """Pad the leading (path) axis of ``x`` to ``num_paths`` entries with
    ``value``, or by repeating the last entry if ``value`` is None."""
    pad = [(0, num_paths - x.shape[0])] + [(0, 0)] * (x.ndim - 1)
    if value is None:
        return np.pad(x, pad, mode="edge")
    return np.pad(x, pad, mode="constant", constant_values=value)


def _select(table, idx, num_links):
    """Row of ``table`` [K, ...] for every link: gathered with the profile
    index ``idx`` [L], or the single row broadcast to L rows if ``idx`` is
    None (K = 1)."""
    if idx is None:
        shape = tf.concat([[num_links], tf.shape(table)[1:]], axis=0)
        return tf.broadcast_to(table, shape)
    return tf.gather(table, idx, axis=0)


def _draw_links(num_links, num_profiles, ds_min, ds_max):
    """Profile index [L] (None if there is a single profile) and delay
    spread [L] [s] of every link, both uniform."""
    rng = sionna.config.tf_rng
    idx = None
    if num_profiles > 1:
        idx = rng.uniform([num_links], minval=0, maxval=num_profiles,
                          dtype=tf.int32)
    ds = rng.uniform([num_links], minval=ds_min, maxval=ds_max,
                     dtype=tf.float32)
    return idx, ds


def _unfold_users(a, tau, batch_size, num_users, link_dims):
    """Links of a user-major flat batch back to users.

    a   : [B*U, 1, Nr, 1, Nt, M, T] -> [B, 1, Nr, U, Nt, M, T]
    tau : [B*U, 1, 1, M]            -> [B, 1, U, M]

    ``link_dims`` = (Nr, Nt, M, T) as Python ints, so that every dimension
    but the batch stays static.
    """
    nr, nt, m, t = link_dims
    a = tf.reshape(a, [batch_size, num_users, nr, nt, m, t])
    a = tf.expand_dims(tf.transpose(a, [0, 2, 1, 3, 4, 5]), axis=1)
    tau = tf.reshape(tau, [batch_size, 1, num_users, m])
    return a, tau


def _cir_to_ofdm(a, tau, frequencies, normalize):
    """Channel frequency response of a CIR,
    h[..., t, f] = sum_m a[..., m, t] exp(-j 2 pi f tau_m).

    Same result as sionna's cir_to_ofdm_channel, written as a contraction over
    the paths so that the [..., M, T, F] product is never materialized.

    a : [B, 1, Nr, U, Nt, M, T], tau : [B, 1, U, M], frequencies : [F]
    Returns [B, 1, Nr, U, Nt, T, F].
    """
    phase = -2. * PI * frequencies * tau[:, 0, :, :, tf.newaxis]   # [B,U,M,F]
    e = tf.exp(tf.complex(tf.zeros_like(phase), phase))
    h = tf.einsum("bxrunmt,bumf->bxruntf", a, e)
    if normalize:
        # Unit average energy per resource element for every sample and link
        c = tf.reduce_mean(tf.square(tf.abs(h)), axis=(2, 4, 5, 6),
                           keepdims=True)
        c = tf.complex(tf.sqrt(c), tf.zeros_like(c))
        h = tf.math.divide_no_nan(h, c)
    return h


def _grid_constants(resource_grid):
    """OFDM symbols, sampling frequency and subcarrier frequencies [F]."""
    num_time_steps = resource_grid.num_ofdm_symbols
    sampling_frequency = 1. / resource_grid.ofdm_symbol_duration
    frequencies = subcarrier_frequencies(resource_grid.fft_size,
                                         resource_grid.subcarrier_spacing,
                                         tf.complex64)
    return num_time_steps, sampling_frequency, frequencies


###########################################################################
# TDL (sum of sinusoids, as the fork's TDL)
###########################################################################

def _sum_of_sinusoids(doppler, theta, phi, alpha_const, t):
    """Unit-power sum-of-sinusoids fading of every tap [L,1,Nr,1,Nt,M,T]
    (the fork's TDL, tdl.py:455-462).

    doppler : [L,1,1,1,1,1,1,1] maximum radian Doppler shift
    theta   : [L,1,1,1,1,M,1,N] angle-of-arrival offsets
    phi     : [L,1,Nr,1,Nt,M,1,N] phases
    t       : [1,1,1,1,1,1,T,1] sample times
    """
    alpha = alpha_const + theta
    argument = doppler * t * tf.cos(alpha) + phi
    h = tf.complex(tf.cos(argument), tf.sin(argument))
    num_sinusoids = tf.cast(tf.shape(theta)[-1], tf.float32)
    scale = tf.complex(1. / tf.sqrt(num_sinusoids), tf.constant(0., tf.float32))
    return scale * tf.reduce_sum(h, axis=-1)


def _add_specular(a, doppler, phi0, los_power, t, los_angle_of_arrival):
    """Add the LoS (specular) component of power ``los_power`` [L] to the
    first tap (the fork's TDL, tdl.py:469-494). Links with zero LoS power
    are unchanged."""
    arg = (tf.squeeze(doppler, axis=-1) * tf.squeeze(t, axis=-1)
           * tf.cos(los_angle_of_arrival) + phi0)                # [L,1,...,T]
    h_spec = tf.complex(tf.cos(arg), tf.sin(arg))
    amp = tf.complex(tf.sqrt(los_power), tf.zeros_like(los_power))
    amp = tf.reshape(amp, [-1, 1, 1, 1, 1, 1, 1])
    return tf.concat([h_spec * amp + a[:, :, :, :, :, :1, :],
                      a[:, :, :, :, :, 1:, :]], axis=5)


def _correlate(a, rx_sqrt, tx_sqrt):
    """Kronecker antenna correlation sqrt(R_rx) a sqrt(R_tx)
    (the fork's TDL, tdl.py:516-523)."""
    a = tf.transpose(a, [0, 1, 3, 5, 6, 2, 4])              # [...,Nr,Nt]
    if rx_sqrt is not None:
        a = tf.matmul(rx_sqrt, a)
    if tx_sqrt is not None:
        a = tf.matmul(a, tx_sqrt)
    return tf.transpose(a, [0, 1, 5, 2, 6, 3, 4])


class MultiTDLChannel(Layer):
    r"""TDL channel with an independent random profile per sample and user.

    Every (sample, user) link draws a TR 38.901 TDL profile uniformly from
    ``models``, a delay spread uniformly from [``delay_spread_min``,
    ``delay_spread_max``] and a UT speed uniformly from [``min_speed``,
    ``max_speed``]. The taps are generated as in Sionna's TDL (sum of
    sinusoids, Rician first tap for the LoS profiles D and E).

    Parameters
    ----------
    carrier_frequency : float
        Carrier frequency [Hz].
    resource_grid : ResourceGrid
        Resource grid of the simulation.
    num_rx_ant : int
        Number of receive antennas.
    num_tx_ant : int
        Number of transmit antennas per user.
    models : list of str or None
        TDL profiles to draw from with equal probability, each one of
        "A".."E". None selects all of them.
    min_speed, max_speed : float
        UT speed range [m/s].
    delay_spread_min, delay_spread_max : float
        Delay spread range [ns].
    correlation : "low" | "medium" | "high"
        Antenna correlation as in TR 38.901 / NTDLChannel. "low" means
        uncorrelated antennas.
    norm_channel : bool
        If True, every link is normalized to unit average energy per
        resource element.
    num_sinusoids : int
        Number of sinusoids of the fading generator.

    Input
    -----
    (x, no) :
        x : [batch size, num_tx, num_tx_ant, num_ofdm_symbols, fft_size],
        tf.complex64. Channel inputs.
        no : Scalar or Tensor, tf.float32. Noise power.

    Output
    ------
    y : [batch size, 1, num_rx_ant, num_ofdm_symbols, fft_size], tf.complex64
        Channel outputs.
    h_freq : [batch size, 1, num_rx_ant, num_tx, num_tx_ant,
              num_ofdm_symbols, fft_size], tf.complex64
        Channel frequency responses.
    """

    def __init__(self,
                 carrier_frequency,
                 resource_grid,
                 num_rx_ant,
                 num_tx_ant,
                 models=None,
                 min_speed=0.,
                 max_speed=0.,
                 delay_spread_min=10.,
                 delay_spread_max=300.,
                 correlation="low",
                 norm_channel=False,
                 num_sinusoids=20):
        super().__init__()

        assert correlation in ("low", "medium", "high")
        assert 0. <= min_speed <= max_speed, "Need 0 <= min_speed <= max_speed"
        assert 0. <= delay_spread_min <= delay_spread_max, \
            "Need 0 <= delay_spread_min <= delay_spread_max"

        models = _profile_set(models)
        self._models = models
        print(f"\n\tLoading MultiTDL {models}: profile, delay spread "
              f"and speed drawn per sample and user, {correlation} "
              f"correlation.\n")

        # Profile constants, read from the fork's TDL so that the LoS split
        # and the power normalization are exactly Sionna's
        delays, powers, los_powers = [], [], []
        for model in models:
            tdl = TDL(model, 100e-9, carrier_frequency)
            d = tdl._delays.numpy()                       # normalized by DS
            p = np.real(tdl._mean_powers.numpy())
            assert len(d) == len(p) == int(tdl._num_clusters)
            delays.append(d)
            powers.append(p)
            los_powers.append(float(np.real(tdl._los_power.numpy()))
                              if tdl.los else 0.)
        num_paths = max(len(d) for d in delays)
        self._num_paths = num_paths
        self._delays = tf.constant(
            np.stack([_pad_paths(d, num_paths, 0.) for d in delays]),
            tf.float32)                                            # [K, M]
        self._powers = tf.constant(
            np.stack([_pad_paths(p, num_paths, 0.) for p in powers]),
            tf.float32)                                            # [K, M]
        self._los_powers = tf.constant(los_powers, tf.float32)    # [K]
        self._has_los = any(lp > 0. for lp in los_powers)

        self._num_rx_ant = num_rx_ant
        self._num_tx_ant = num_tx_ant
        self._num_sinusoids = num_sinusoids
        self._ds_min = delay_spread_min * 1e-9
        self._ds_max = delay_spread_max * 1e-9
        self._norm_channel = norm_channel

        # Maximum radian Doppler shifts, as the fork's TDL._compute_doppler
        self._min_doppler = 2. * PI * min_speed / SPEED_OF_LIGHT * carrier_frequency
        self._max_doppler = 2. * PI * max_speed / SPEED_OF_LIGHT * carrier_frequency
        self._los_angle_of_arrival = tf.constant(PI / 4., tf.float32)
        alpha_const = 2. * PI / num_sinusoids * tf.range(
            1., num_sinusoids + 1, 1., dtype=tf.float32)
        self._alpha_const = tf.reshape(alpha_const, [1, 1, 1, 1, 1, 1, 1, -1])

        # Kronecker correlation, as NTDLChannel. "low" is uncorrelated.
        self._rx_sqrt = self._tx_sqrt = None
        if correlation != "low":
            alpha, beta = (0.9, 0.3) if correlation == "medium" else (0.9, 0.9)
            self._rx_sqrt = expand_to_rank(
                matrix_sqrt(exp_corr_mat(alpha, num_rx_ant)), 7, 0)
            self._tx_sqrt = expand_to_rank(
                matrix_sqrt(exp_corr_mat(beta, num_tx_ant)), 7, 0)

        (self._num_time_steps, self._sampling_frequency,
         self._frequencies) = _grid_constants(resource_grid)
        t = tf.range(self._num_time_steps, dtype=tf.float32) / self._sampling_frequency
        self._t = tf.reshape(t, [1, 1, 1, 1, 1, 1, -1, 1])
        self._link_dims = (num_rx_ant, num_tx_ant, num_paths,
                           self._num_time_steps)

        self._apply_channel = ApplyOFDMChannel()
        _ = sionna.config.tf_rng                    # create the RNG eagerly

    def _cir(self, idx, ds):
        """CIR of L links with profile index ``idx`` [L] (None if K = 1) and
        delay spread ``ds`` [L] [s].

        Returns a [L, 1, Nr, 1, Nt, M, T] and tau [L, 1, 1, M].
        """
        num_links = tf.shape(ds)[0]
        m, n = self._num_paths, self._num_sinusoids
        rng = sionna.config.tf_rng

        doppler = rng.uniform([num_links, 1, 1, 1, 1, 1, 1, 1],
                              self._min_doppler, self._max_doppler)
        theta = rng.uniform([num_links, 1, 1, 1, 1, m, 1, n], -PI / n, PI / n)
        phi = rng.uniform([num_links, 1, self._num_rx_ant, 1,
                           self._num_tx_ant, m, 1, n], -PI, PI)
        a = _sum_of_sinusoids(doppler, theta, phi, self._alpha_const, self._t)

        powers = _select(self._powers, idx, num_links)              # [L, M]
        amp = tf.complex(tf.sqrt(powers), tf.zeros_like(powers))
        a = a * tf.reshape(amp, [-1, 1, 1, 1, 1, m, 1])

        if self._has_los:
            phi0 = rng.uniform([num_links, 1, 1, 1, 1, 1, 1], -PI, PI)
            a = _add_specular(a, doppler, phi0,
                              _select(self._los_powers, idx, num_links),
                              self._t, self._los_angle_of_arrival)

        if self._rx_sqrt is not None:
            a = _correlate(a, self._rx_sqrt, self._tx_sqrt)

        tau = _select(self._delays, idx, num_links) * ds[:, tf.newaxis]
        tau = tf.reshape(tau, [-1, 1, 1, m])
        return tf.stop_gradient(a), tf.stop_gradient(tau)

    def call(self, inputs):
        x, no = inputs
        batch_size = tf.shape(x)[0]
        num_users = x.shape[1]
        if num_users is None:
            raise ValueError("MultiTDLChannel needs a static number of users.")

        idx, ds = _draw_links(batch_size * num_users, len(self._models),
                              self._ds_min, self._ds_max)
        a, tau = self._cir(idx, ds)
        a, tau = _unfold_users(a, tau, batch_size, num_users,
                               self._link_dims)
        h = _cir_to_ofdm(a, tau, self._frequencies, self._norm_channel)
        y = self._apply_channel([x, h, no])
        return y, h


###########################################################################
# CDL (geometric, as the fork's CDL)
###########################################################################

class MultiCDLChannel(Layer):
    r"""CDL channel with an independent random profile per sample and user.

    Every (sample, user) link draws a TR 38.901 CDL profile uniformly from
    ``models``, a delay spread uniformly from [``delay_spread_min``,
    ``delay_spread_max``] and a UT velocity as Sionna's CDL (speed uniform in
    [``min_speed``, ``max_speed``], random direction). The links are uplink
    and are generated with the fork's ChannelCoefficientsGenerator; LoS
    (CDL-D/E) and NLoS links share one batch.

    Parameters
    ----------
    carrier_frequency : float
        Carrier frequency [Hz].
    resource_grid : ResourceGrid
        Resource grid of the simulation.
    bs_array : PanelArray
        Base station (receive) antenna array.
    ut_array : PanelArray
        User terminal (transmit) antenna array.
    models : list of str or None
        CDL profiles to draw from with equal probability, each one of
        "A".."E". None selects all of them.
    min_speed, max_speed : float
        UT speed range [m/s].
    delay_spread_min, delay_spread_max : float
        Delay spread range [ns].
    spread_ut_orientation : bool
        If True and there are several users, user slot u always faces
        azimuth pi + 2 pi u / num_tx (NCDLChannel instead spreads the
        azimuths over its tiled profile list and reassigns them to the users
        at random every batch). Otherwise, and for a single user, every UT
        has the Sionna default azimuth pi (facing the BS).
    norm_channel : bool
        If True, every link is normalized to unit average energy per
        resource element.

    Input
    -----
    (x, no) :
        x : [batch size, num_tx, num_tx_ant, num_ofdm_symbols, fft_size],
        tf.complex64. Channel inputs.
        no : Scalar or Tensor, tf.float32. Noise power.

    Output
    ------
    y : [batch size, 1, num_rx_ant, num_ofdm_symbols, fft_size], tf.complex64
        Channel outputs.
    h_freq : [batch size, 1, num_rx_ant, num_tx, num_tx_ant,
              num_ofdm_symbols, fft_size], tf.complex64
        Channel frequency responses.
    """

    def __init__(self,
                 carrier_frequency,
                 resource_grid,
                 bs_array,
                 ut_array,
                 models=None,
                 min_speed=0.,
                 max_speed=0.,
                 delay_spread_min=10.,
                 delay_spread_max=300.,
                 spread_ut_orientation=True,
                 norm_channel=False):
        super().__init__()

        assert 0. <= min_speed <= max_speed, "Need 0 <= min_speed <= max_speed"
        assert 0. <= delay_spread_min <= delay_spread_max, \
            "Need 0 <= delay_spread_min <= delay_spread_max"

        models = _profile_set(models)
        self._models = models
        print(f"\n\tLoading MultiCDL {models}: profile, delay spread "
              f"and velocity drawn per sample and user.\n")

        # Profile constants after the fork's processing (ray offsets, LoS
        # split, uplink angle swap). The delay spread passed here is unused.
        cdls = [CDL(model, 1.0, carrier_frequency, ut_array, bs_array,
                    "uplink", min_speed=min_speed, max_speed=max_speed)
                for model in models]
        num_paths = max(int(c.num_clusters) for c in cdls)
        self._num_paths = num_paths

        def per_path(attr, pad):
            """[K, M(, rays)] stack of a per-cluster constant."""
            return tf.constant(np.stack(
                [_pad_paths(getattr(c, attr).numpy()[0, 0, 0], num_paths, pad)
                 for c in cdls]), tf.float32)

        def per_profile(attr):
            """[K] stack of a per-profile scalar."""
            return tf.constant(
                [getattr(c, attr).numpy()[0, 0, 0] for c in cdls], tf.float32)

        # Padded clusters: zero power, finite angles and a positive XPR
        # (the generator scales by sqrt(1/xpr)), and a delay above every
        # real cluster so that they sort last (the LoS ray is added to the
        # first cluster after sorting by delay).
        self._delays = tf.constant(np.stack(
            [_pad_paths(d, num_paths, d.max() + 1.)
             for d in (c._delays.numpy()[0, 0, 0] for c in cdls)]),
            tf.float32)                                            # [K, M]
        self._powers = per_path("_powers", 0.)                     # [K, M]
        self._aoa = per_path("_aoa", None)                         # [K, M, 20]
        self._aod = per_path("_aod", None)
        self._zoa = per_path("_zoa", None)
        self._zod = per_path("_zod", None)
        self._xpr = per_path("_xpr", None)
        self._k_factor = per_profile("_k_factor")                  # [K]
        self._los_aoa = per_profile("_los_aoa")
        self._los_aod = per_profile("_los_aod")
        self._los_zoa = per_profile("_los_zoa")
        self._los_zod = per_profile("_los_zod")
        self._los = tf.constant([bool(c.los) for c in cdls], tf.bool)

        # One CDL instance provides the coefficient generator (shared arrays)
        # and the random intra-cluster ray coupling
        self._cdl = cdls[0]

        self._min_speed = float(min_speed)
        self._max_speed = float(max_speed)
        self._ds_min = delay_spread_min * 1e-9
        self._ds_max = delay_spread_max * 1e-9
        self._spread_ut_orientation = spread_ut_orientation
        self._norm_channel = norm_channel

        (self._num_time_steps, self._sampling_frequency,
         self._frequencies) = _grid_constants(resource_grid)
        self._link_dims = (bs_array.num_ant, ut_array.num_ant, num_paths,
                           self._num_time_steps)

        self._apply_channel = ApplyOFDMChannel()
        _ = sionna.config.tf_rng                    # create the RNG eagerly

    def _ut_orientations(self, num_users):
        """[U, 3] UT orientation of every user slot."""
        azimuth = np.full([num_users], PI)
        if self._spread_ut_orientation and num_users > 1:
            azimuth = PI + 2. * PI * np.arange(num_users) / num_users
        zeros = np.zeros([num_users])
        return tf.constant(np.stack([azimuth, zeros, zeros], axis=-1),
                           tf.float32)

    def _cir(self, idx, ds, ut_orientations):
        """CIR of L links with profile index ``idx`` [L] (None if K = 1),
        delay spread ``ds`` [L] [s] and UT orientation ``ut_orientations``
        [L, 3], drawn as the fork's CDL.__call__.

        Returns a [L, 1, Nr, 1, Nt, M, T] and tau [L, 1, 1, M].
        """
        num_links = tf.shape(ds)[0]
        rng = sionna.config.tf_rng

        def per_link(table):
            return tf.reshape(_select(table, idx, num_links), [-1, 1, 1])

        # UT velocities
        v_r = rng.uniform([num_links, 1], minval=self._min_speed,
                          maxval=self._max_speed, dtype=tf.float32)
        v_phi = rng.uniform([num_links, 1], minval=0., maxval=2. * PI,
                            dtype=tf.float32)
        v_theta = rng.uniform([num_links, 1], minval=0., maxval=PI,
                              dtype=tf.float32)
        velocities = tf.stack([v_r * cos(v_phi) * sin(v_theta),
                               v_r * sin(v_phi) * sin(v_theta),
                               v_r * cos(v_theta)], axis=-1)        # [L,1,3]

        topology = Topology(velocities=velocities,
                            moving_end="tx",
                            los_aoa=per_link(self._los_aoa),
                            los_aod=per_link(self._los_aod),
                            los_zoa=per_link(self._los_zoa),
                            los_zod=per_link(self._los_zod),
                            los=per_link(self._los),
                            distance_3d=tf.zeros([num_links, 1, 1], tf.float32),
                            tx_orientations=ut_orientations[:, tf.newaxis],
                            rx_orientations=tf.zeros([num_links, 1, 3],
                                                     tf.float32))

        # Cluster constants of every link, then the random coupling of the
        # rays within each cluster (TR 38.901 step 8)
        delays = _select(self._delays, idx, num_links) * ds[:, tf.newaxis]
        delays = tf.reshape(delays, [-1, 1, 1, self._num_paths])
        powers = tf.reshape(_select(self._powers, idx, num_links),
                            [-1, 1, 1, self._num_paths])
        aoa, aod, zoa, zod, xpr = [
            _select(table, idx, num_links)[:, tf.newaxis, tf.newaxis]
            for table in (self._aoa, self._aod, self._zoa, self._zod,
                          self._xpr)]                         # [L,1,1,M,20]
        aoa, aod, zoa, zod = [self._cdl._shuffle_angles(v)
                              for v in (aoa, aod, zoa, zod)]

        # No ray or cluster is ever removed (padded clusters have zero power)
        rays = Rays(delays=delays, powers=powers, aoa=aoa, aod=aod, zoa=zoa,
                    zod=zod, xpr=xpr, ray_mask=tf.zeros_like(aoa),
                    cluster_mask=tf.zeros_like(powers))

        h, tau = self._cdl._cir_sampler(self._num_time_steps,
                                        self._sampling_frequency,
                                        per_link(self._k_factor), rays,
                                        topology)
        a = tf.transpose(h, [0, 2, 4, 1, 5, 3, 6])
        tau = tf.transpose(tau, [0, 2, 1, 3])
        return tf.stop_gradient(a), tf.stop_gradient(tau)

    def call(self, inputs):
        x, no = inputs
        batch_size = tf.shape(x)[0]
        num_users = x.shape[1]
        if num_users is None:
            raise ValueError("MultiCDLChannel needs a static number of users.")

        idx, ds = _draw_links(batch_size * num_users, len(self._models),
                              self._ds_min, self._ds_max)
        ut_orientations = tf.tile(self._ut_orientations(num_users),
                                  [batch_size, 1])                   # [L, 3]
        a, tau = self._cir(idx, ds, ut_orientations)
        a, tau = _unfold_users(a, tau, batch_size, num_users,
                               self._link_dims)
        h = _cir_to_ofdm(a, tau, self._frequencies, self._norm_channel)
        y = self._apply_channel([x, h, no])
        return y, h
