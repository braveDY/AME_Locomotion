# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom Go2 rough-terrain velocity-tracking environment configuration.

This module intentionally does not inherit from ``LocomotionVelocityRoughEnvCfg``.
Keep task-specific changes here so they do not affect the shared locomotion template.
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from ame_locomotion import mdp

import isaaclab.terrains as terrain_gen
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


@configclass
class CustomUnitreeGo2SceneCfg(InteractiveSceneCfg):
    """Scene definition for the custom Go2 task."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=(
                f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/"
                "TilesMarbleSpiderWhiteBrickBondHoned.mdl"
            ),
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class CustomUnitreeGo2CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class CustomUnitreeGo2ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)


@configclass
class CustomUnitreeGo2ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-2.0, n_max=2.0))
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.elevation_map,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "noise": True},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.elevation_map,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "noise": False},
        )

    critic: CriticCfg = CriticCfg()


@configclass
class CustomUnitreeGo2EventsCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (0.0, 0.0),
        },
    )


@configclass
class CustomUnitreeGo2RewardsCfg:
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.75,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-0.0002)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.01,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )


@configclass
class CustomUnitreeGo2TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )


@configclass
class CustomUnitreeGo2CurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class UnitreeGo2CustomEnvCfg(ManagerBasedRLEnvCfg):
    """Independent Go2 rough-terrain velocity-tracking environment."""

    scene: CustomUnitreeGo2SceneCfg = CustomUnitreeGo2SceneCfg(num_envs=4096, env_spacing=2.5)
    observations: CustomUnitreeGo2ObservationsCfg = CustomUnitreeGo2ObservationsCfg()
    actions: CustomUnitreeGo2ActionsCfg = CustomUnitreeGo2ActionsCfg()
    commands: CustomUnitreeGo2CommandsCfg = CustomUnitreeGo2CommandsCfg()
    rewards: CustomUnitreeGo2RewardsCfg = CustomUnitreeGo2RewardsCfg()
    terminations: CustomUnitreeGo2TerminationsCfg = CustomUnitreeGo2TerminationsCfg()
    events: CustomUnitreeGo2EventsCfg = CustomUnitreeGo2EventsCfg()
    curriculum: CustomUnitreeGo2CurriculumCfg = CustomUnitreeGo2CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.terrain.terrain_generator.curriculum = True
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01


@configclass
class UnitreeGo2CustomEnvCfg_PLAY(UnitreeGo2CustomEnvCfg):
    """Play configuration for the custom Go2 AME environment."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.terrain_generator.num_rows = 4
        self.scene.terrain.terrain_generator.num_cols = 4
        self.scene.terrain.terrain_generator.size = (8.0, 8.0)

        # Select one terrain by uncommenting its entry and commenting out others.
        self.scene.terrain.terrain_generator.sub_terrains = {
            # === 1. 经典粗糙地形 (ROUGH_TERRAINS_CFG 标准预设) ===
            "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                proportion=1.0, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
            ),  # 随机起伏粗糙地面 (Continuous Rough Terrain)
            # "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            #     proportion=1.0, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
            # ),  # 离散随机高低方块网格 (Discrete Boxes Grid)
            # "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            #     proportion=1.0, step_height_range=(0.05, 0.23), step_width=0.3, platform_width=3.0, border_width=1.0, holes=False
            # ),  # 正金字塔台阶/楼梯 (Upward Pyramid Stairs)
            # "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            #     proportion=1.0, step_height_range=(0.05, 0.23), step_width=0.3, platform_width=3.0, border_width=1.0, holes=False
            # ),  # 倒金字塔下台阶/楼梯 (Downward Inverted Stairs)
            # "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            #     proportion=1.0, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
            # ),  # 正金字塔连续斜坡 (Pyramid Slopes)
            # "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            #     proportion=1.0, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
            # ),  # 倒金字塔下斜坡/碗状坡地 (Inverted Pyramid Slopes)

            # === 2. 官方三角网格地形 (Official Trimesh Terrains) ===
            # "mesh_plane": terrain_gen.MeshPlaneTerrainCfg(
            #     proportion=1.0
            # ),  # 纯平坦地面 (Flat Plane)
            # "mesh_gap": terrain_gen.MeshGapTerrainCfg(
            #     proportion=1.0, gap_width_range=(0.3, 0.6), platform_width=2.0
            # ),  # 环绕深沟/裂缝跨越地形 (Gap / Trench)
            # "mesh_pit": terrain_gen.MeshPitTerrainCfg(
            #     proportion=1.0, pit_depth_range=(0.1, 0.3), platform_width=2.0, double_pit=False
            # ),  # 凹坑下潜与出坑地形 (Pit / Depression)
            # "mesh_box": terrain_gen.MeshBoxTerrainCfg(
            #     proportion=1.0, box_height_range=(0.1, 0.3), platform_width=2.0, double_box=False
            # ),  # 箱体高台地形 (Box Platform)
            # "mesh_rails": terrain_gen.MeshRailsTerrainCfg(
            #     proportion=1.0, rail_thickness_range=(0.1, 0.3), rail_height_range=(0.05, 0.2), platform_width=2.0
            # ),  # 水平导轨/障碍条 (Horizontal Rails)
            # "mesh_floating_ring": terrain_gen.MeshFloatingRingTerrainCfg(
            #     proportion=1.0, ring_width_range=(0.3, 0.5), ring_height_range=(0.1, 0.2), ring_thickness=0.1, platform_width=2.0
            # ),  # 悬浮环形几何障碍 (Floating Ring)
            # "mesh_star": terrain_gen.MeshStarTerrainCfg(
            #     proportion=1.0, num_bars=5, bar_width_range=(0.2, 0.4), bar_height_range=(0.05, 0.2), platform_width=2.0
            # ),  # 星形放射状障碍地形 (Star Pattern)
            # "mesh_repeated_pyramids": terrain_gen.MeshRepeatedPyramidsTerrainCfg(
            #     proportion=1.0, platform_width=2.0,
            #     object_params_start=terrain_gen.MeshRepeatedPyramidsTerrainCfg.ObjectCfg(num_objects=30, height=0.1, radius=0.2),
            #     object_params_end=terrain_gen.MeshRepeatedPyramidsTerrainCfg.ObjectCfg(num_objects=50, height=0.2, radius=0.3),
            # ),  # 密集圆锥/尖锥立桩群 (Repeated Pyramids)
            # "mesh_repeated_boxes": terrain_gen.MeshRepeatedBoxesTerrainCfg(
            #     proportion=1.0, platform_width=2.0,
            #     object_params_start=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(num_objects=30, height=0.1, size=(0.3, 0.3)),
            #     object_params_end=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(num_objects=50, height=0.2, size=(0.4, 0.4)),
            # ),  # 密集斜放方块群 (Repeated Boxes)
            # "mesh_repeated_cylinders": terrain_gen.MeshRepeatedCylindersTerrainCfg(
            #     proportion=1.0, platform_width=2.0,
            #     object_params_start=terrain_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(num_objects=30, height=0.1, radius=0.2),
            #     object_params_end=terrain_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(num_objects=50, height=0.2, radius=0.3),
            # ),  # 密集圆柱立桩群 (Repeated Cylinders)

            # === 3. 官方高度图地形 (Official HeightField Terrains) ===
            # "hf_steppingstones": terrain_gen.HfSteppingStonesTerrainCfg(
            #     proportion=1.0, stone_height_max=0.05, stone_width_range=(0.25, 0.5),
            #     stone_distance_range=(0.05, 0.25), platform_width=2.0, holes_depth=-2.0, border_width=0.25
            # ),  # 梅花桩/跳岩石块地形 (Stepping Stones)
            # "hf_wave": terrain_gen.HfWaveTerrainCfg(
            #     proportion=1.0, amplitude_range=(0.1, 0.3), num_waves=4, border_width=0.25
            # ),  # 正弦连续波浪起伏地面 (Wave Terrain)
            # "hf_discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            #     proportion=1.0, obstacle_width_range=(0.3, 0.6), obstacle_height_range=(0.05, 0.15),
            #     num_obstacles=40, platform_width=2.0, border_width=0.25
            # ),  # 离散凸起方块障碍 (Discrete Obstacles)
        }
        self.observations.policy.enable_corruption = False
        self.observations.policy.height_scan.params["noise"] = False
        self.events.base_external_force_torque = None
