__author__ = "Antoine Richard, Matteo El Hariry"
__copyright__ = (
    "Copyright 2023, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Antoine Richard"
__email__ = "antoine.richard@uni.lu"
__status__ = "development"

from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.heron import (
    Heron,
)
from omniisaacgymenvs.robots.articulations.views.heron_view import (
    HeronView,
)
from omniisaacgymenvs.utils.pin import VisualPin
from omniisaacgymenvs.utils.arrow import VisualArrow

from omniisaacgymenvs.tasks.USV.USV_task_factory import (
    task_factory,
)
from omniisaacgymenvs.tasks.USV.USV_core import parse_data_dict
from omniisaacgymenvs.tasks.USV.USV_task_rewards import (
    Penalties,
)
from omniisaacgymenvs.tasks.USV.USV_disturbances import (
    ForceDisturbance,
    TorqueDisturbance,
    NoisyObservations,
    NoisyActions,
    MassDistributionDisturbances,
)

from omniisaacgymenvs.envs.USV.Hydrodynamics import *
from omniisaacgymenvs.envs.USV.Hydrostatics import *
from omniisaacgymenvs.envs.USV.ThrusterDynamics import *

from omni.isaac.core.utils.torch.rotations import *
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.prims.xform_prim import XFormPrim

from typing import Dict, List, Tuple

import numpy as np
import omni
import time
import math
import torch
from gym import spaces
from dataclasses import dataclass

EPS = 1e-6  # small constant to avoid divisions by 0 and log(0)


class Berth:
    """
    泊位类，创建三面墙的长方形泊位
    不参与碰撞，只用于可视化和奖励
    """
    
    def __init__(self, prim_path: str, target_position: torch.Tensor, 
                 boat_length: float = 1.3, boat_width: float = 1.0,
                 wall_thickness: float = 0.10, wall_height: float = 1.0, 
                 z_base: float = 0.0, alpha: float = 0.3):
        """
        初始化泊位
        
        Args:
            prim_path: USD路径
            target_position: 目标点位置（泊位中心）
            boat_length: 船长
            boat_width: 船宽
            wall_thickness: 墙厚
            wall_height: 墙高
            z_base: 墙基座高度
            alpha: 透明度
        """
        self.prim_path = prim_path
        self.target_position = target_position
        self.boat_length = boat_length
        self.boat_width = boat_width
        self.wall_thickness = wall_thickness
        self.wall_height = wall_height
        self.z_base = z_base
        self.alpha = alpha
        
        # 泊位尺寸（基于船尺寸）
        self.L_berth = 1.4 * boat_length  # 泊位长度
        self.W_berth = 1.6 * boat_width   # 泊位宽度
        
        # 确保泊位尺寸是正数
        self.L_berth = max(self.L_berth, 0.1)
        self.W_berth = max(self.W_berth, 0.1)
        
        # 泊位朝向（初始化即随机，避免"post_reset太早改、可视化没跟上"的问题）
        self.berth_orientation = float(torch.rand(()) * 2 * math.pi)
        
        # 创建泊位墙体
        self.create_berth_walls()
    
    def create_berth_walls(self):
        """创建泊位三面墙：侧墙与后墙内侧齐平，不外伸"""
        from omniisaacgymenvs.utils.shape_utils import createPrim, createColor, applyMaterial, applyTransforms
        from pxr import UsdGeom

        stage = omni.usd.get_context().get_stage()

        # 颜色
        material_path = f"{self.prim_path}/berth_material"
        material = createColor(stage, material_path, [0.5, 0.8, 1.0])

        L = float(self.L_berth)
        W = float(self.W_berth)
        t = float(self.wall_thickness)
        h = float(self.wall_height)
        zc = self.z_base + 0.5 * h

        # === 侧墙尺寸与位置（长度 L - t，中心 y = +t/2；x 内收 t/2）===
        side_len = max(L - t, 0.01)

        # 以红点为中心
        tx = float(self.target_position[0])
        ty = float(self.target_position[1])

        # 朝向（围绕Z轴的偏航），注意 quaternion 采用 [x,y,z,w]
        yaw = float(self.berth_orientation)
        s, c = math.sin(0.5 * yaw), math.cos(0.5 * yaw)
        rot_q = [0.0, 0.0, s, c]

        # 局部->世界（先旋转，再平移到红点）
        def to_world(px, py, pz):
            x = tx + px * math.cos(yaw) - py * math.sin(yaw)
            y = ty + px * math.sin(yaw) + py * math.cos(yaw)
            return [x, y, pz]

        # 左墙
        left_path = f"{self.prim_path}/left_wall"
        _, left_prim = createPrim(left_path, name="", geom_type=UsdGeom.Cube)
        UsdGeom.Cube(left_prim).CreateSizeAttr(1.0)            # 避免 scale*2
        left_x = -(W * 0.5 - 0.5 * t)
        left_y =  t * 0.5
        applyTransforms(left_prim, to_world(left_x, left_y, zc), rot_q, [t, side_len, h])
        applyMaterial(left_prim, material)

        # 右墙
        right_path = f"{self.prim_path}/right_wall"
        _, right_prim = createPrim(right_path, name="", geom_type=UsdGeom.Cube)
        UsdGeom.Cube(right_prim).CreateSizeAttr(1.0)
        right_x =  (W * 0.5 - 0.5 * t)
        right_y =  t * 0.5
        applyTransforms(right_prim, to_world(right_x, right_y, zc), rot_q, [t, side_len, h])
        applyMaterial(right_prim, material)

        # === 后墙尺寸与位置（宽度 W - 2t，中心 y = -L/2 + t/2）===
        back_path = f"{self.prim_path}/back_wall"
        _, back_prim = createPrim(back_path, name="", geom_type=UsdGeom.Cube)
        UsdGeom.Cube(back_prim).CreateSizeAttr(1.0)
        back_w = max(W - 2.0 * t, 0.01)
        back_x = 0.0
        back_y = -L * 0.5 + t * 0.5
        applyTransforms(back_prim, to_world(back_x, back_y, zc), rot_q, [back_w, t, h])
        applyMaterial(back_prim, material)

        # 检验
        """
        print(f"[Berth] L={L:.3f} W={W:.3f} t={t:.3f} | "
              f"Lx={left_x:.3f} Rx={right_x:.3f} | side_len={side_len:.3f} "
              f"| back_w={back_w:.3f} back_y={back_y:.3f}")"""
        
       

    def set_yaw(self, yaw: float):
        """更新泊位朝向并同步三面墙的变换。"""
        self.berth_orientation = float(yaw)
        self._update_walls_transform()

    def _update_walls_transform(self):
        from omniisaacgymenvs.utils.shape_utils import applyTransforms
        from omni.isaac.core.utils.prims import get_prim_at_path
        import math

        L = float(self.L_berth); W = float(self.W_berth)
        t = float(self.wall_thickness); h = float(self.wall_height)
        zc = self.z_base + 0.5 * h
        tx, ty = float(self.target_position[0]), float(self.target_position[1])

        yaw = float(self.berth_orientation)
        s, c = math.sin(0.5 * yaw), math.cos(0.5 * yaw)
        rot_q = [0.0, 0.0, s, c]

        # 把"泊位局部 (px,py)"先绕 Z 旋转 yaw，再平移到 env 局部 (tx,ty)
        def to_env(px, py, pz):
            x = tx + px * math.cos(yaw) - py * math.sin(yaw)
            y = ty + px * math.sin(yaw) + py * math.cos(yaw)
            return [x, y, pz]

        # 句柄，不存在就重建一次
        left  = get_prim_at_path(f"{self.prim_path}/left_wall")
        right = get_prim_at_path(f"{self.prim_path}/right_wall")
        back  = get_prim_at_path(f"{self.prim_path}/back_wall")
        if not left or not right or not back:
            self.create_berth_walls()
            left  = get_prim_at_path(f"{self.prim_path}/left_wall")
            right = get_prim_at_path(f"{self.prim_path}/right_wall")
            back  = get_prim_at_path(f"{self.prim_path}/back_wall")

        # 尺寸（与创建时一致）
        side_len = max(L - t, 0.01)
        back_w   = max(W - 2.0 * t, 0.01)

        # 左/右/后墙的 env 局部中心点
        left_x,  left_y  = -(W*0.5 - 0.5*t),  0.5*t
        right_x, right_y =  (W*0.5 - 0.5*t),  0.5*t
        back_x,  back_y  =  0.0,             -0.5*L + 0.5*t

        # 应用本地位姿与缩放（本地=env 局部）
        applyTransforms(left,  to_env(left_x,  left_y,  zc), rot_q, [t, side_len, h])
        applyTransforms(right, to_env(right_x, right_y, zc), rot_q, [t, side_len, h])
        applyTransforms(back,  to_env(back_x,  back_y,  zc), rot_q, [back_w, t, h])


class USVVirtual(RLTask):
    """
    The main class used to run tasks on the floating platform.
    Unlike other class in this repo, this class can be used to run different tasks.
    The idea being to extend it to multitask RL in the future."""

    def __init__(
        self,
        name: str,  # name of the Task
        sim_config,  # SimConfig instance for parsing cfg
        env,  # env instance of VecEnvBase or inherited class
        offset=None,  # transform offset in World
    ) -> None:
        # parse configurations, set task-specific members
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config
        self._heron_cfg = self._task_cfg["env"]["platform"]
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._max_episode_length = self._task_cfg["env"]["maxEpisodeLength"]
        self._discrete_actions = self._task_cfg["env"]["action_mode"]
        self._observation_frame = self._task_cfg["env"]["observation_frame"]
        self._device = self._cfg["sim_device"]
        self.step = 0

        # 调试开关：泊位坐标系统自测
        self.debug_berth = self._task_cfg["env"].get("debug_berth", False)  # 从cfg读取
        
        # 泊位门外扇区出生点配置参数
        self.min_spawn_dist = self._task_cfg["env"].get("min_spawn_dist", 0.3)
        self.max_spawn_dist = self._task_cfg["env"].get("max_spawn_dist", 12.0)
        self.clearance = self._task_cfg["env"].get("clearance", 0.6)
        self.lateral_span = self._task_cfg["env"].get("lateral_span", 0.8)
        self.heading_err_deg = self._task_cfg["env"].get("heading_err_deg", 20.0)

        # Split the maximum amount of thrust across all thrusters.
        self.split_thrust = self._task_cfg["env"]["split_thrust"]

        # Domain randomization and adaptation
        self.UF = ForceDisturbance(
            self._task_cfg["env"]["disturbances"]["forces"],
            self._num_envs,
            self._device,
        )
        self.TD = TorqueDisturbance(
            self._task_cfg["env"]["disturbances"]["torques"],
            self._num_envs,
            self._device,
        )
        self.ON = NoisyObservations(
            self._task_cfg["env"]["disturbances"]["observations"]
        )
        self.AN = NoisyActions(self._task_cfg["env"]["disturbances"]["actions"])
        self.MDD = MassDistributionDisturbances(
            self._task_cfg["env"]["disturbances"]["mass"], self.num_envs, self._device
        )
        # Collects the platform parameters
        self.dt = self._task_cfg["sim"]["dt"]
        # Collects the task parameters
        task_cfg = self._task_cfg["env"]["task_parameters"]
        reward_cfg = self._task_cfg["env"]["reward_parameters"]
        penalty_cfg = self._task_cfg["env"]["penalties_parameters"]

        # physics
        self.gravity = self._task_cfg["sim"]["gravity"][2]
        # Water density kg/m^3
        self.water_density = self._task_cfg["dynamics"]["hydrostatics"]["water_density"]
        self.timeConstant = self._task_cfg["dynamics"]["thrusters"]["timeConstant"]

        # Water Current
        self.use_water_current = self._task_cfg["env"]["water_current"][
            "use_water_current"
        ]
        self.flow_vel = self._task_cfg["env"]["water_current"]["flow_velocity"]

        # Action-safety and current-compensation gains
        as_cfg = (self._task_cfg.get("env", {}) or {}).get("action_safety", {}) or {}
        self.as_g_throttle = float(as_cfg.get("g_throttle", 0.45))
        self.as_g_diff     = float(as_cfg.get("g_diff", 0.75))
        self.as_g_bias     = float(as_cfg.get("g_bias", 0.30))
        self.as_back_buf   = float(as_cfg.get("back_wall_buffer", 0.12))
        self.as_back_push  = float(as_cfg.get("back_wall_push", 0.35))
        self.as_outer_yaw  = float(as_cfg.get("outer_yaw_gain", 0.18))
        self.as_crab_gain  = float(as_cfg.get("current_heading_gain", 0.60))
        self.as_k_vx     = float(as_cfg.get("lateral_vel_gain", 0.60))
        self.as_entrance_guard = float(as_cfg.get("entrance_guard", 0.05))

        # ILOS gains (set ki=0 to disable integral initially)
        ilos_cfg = (self._task_cfg.get("env", {}) or {}).get("ilos", {}) or {}
        self.kp_ilos  = float(ilos_cfg.get("kp", 0.0))
        self.ki_ilos  = float(ilos_cfg.get("ki", 0.0))
        self.ilos_zeta = torch.zeros(self._num_envs, device=self._device)

        # hydrostatics
        self.average_hydrostatics_force_value = self._task_cfg["dynamics"][
            "hydrostatics"
        ]["average_hydrostatics_force_value"]
        self.amplify_torque = self._task_cfg["dynamics"]["hydrostatics"][
            "amplify_torque"
        ]

        # boxes dimension to compute hydrostatic forces and torques
        self.box_density = self._task_cfg["dynamics"]["hydrostatics"][
            "material_density"
        ]
        self.box_width = self._task_cfg["dynamics"]["hydrostatics"]["box_width"]
        self.box_length = self._task_cfg["dynamics"]["hydrostatics"]["box_length"]
        self.waterplane_area = self._task_cfg["dynamics"]["hydrostatics"][
            "waterplane_area"
        ]
        self.heron_zero_height = self._task_cfg["dynamics"]["hydrostatics"][
            "heron_zero_height"
        ]
        self.max_volume = (
            self.box_width * self.box_length * (self.heron_zero_height + 20)
        )  # TODO: Hardcoded value
        self.heron_mass = self._task_cfg["dynamics"]["hydrostatics"]["mass"]

        # thrusters dynamics
        # interpolation
        self.cmd_lower_range = self._task_cfg["dynamics"]["thrusters"][
            "cmd_lower_range"
        ]
        self.cmd_upper_range = self._task_cfg["dynamics"]["thrusters"][
            "cmd_upper_range"
        ]
        self.numberOfPointsForInterpolation = self._task_cfg["dynamics"]["thrusters"][
            "interpolation"
        ]["numberOfPointsForInterpolation"]
        self.interpolationPointsFromRealDataLeft = self._task_cfg["dynamics"][
            "thrusters"
        ]["interpolation"]["interpolationPointsFromRealDataLeft"]
        self.interpolationPointsFromRealDataRight = self._task_cfg["dynamics"][
            "thrusters"
        ]["interpolation"]["interpolationPointsFromRealDataRight"]
        # least square methode
        self.neg_cmd_coeff = self._task_cfg["dynamics"]["thrusters"][
            "leastSquareMethod"
        ]["neg_cmd_coeff"]
        self.pos_cmd_coeff = self._task_cfg["dynamics"]["thrusters"][
            "leastSquareMethod"
        ]["pos_cmd_coeff"]
        # acceleration
        self.alpha = self._task_cfg["dynamics"]["acceleration"]["alpha"]
        self.last_time = self._task_cfg["dynamics"]["acceleration"]["last_time"]
        # hydrodynamics constants
        self.linear_damping = self._task_cfg["dynamics"]["hydrodynamics"][
            "linear_damping"
        ]
        self.quadratic_damping = self._task_cfg["dynamics"]["hydrodynamics"][
            "quadratic_damping"
        ]
        self.linear_damping_forward_speed = self._task_cfg["dynamics"]["hydrodynamics"][
            "linear_damping_forward_speed"
        ]
        self.offset_linear_damping = self._task_cfg["dynamics"]["hydrodynamics"][
            "offset_linear_damping"
        ]
        self.offset_lin_forward_damping_speed = self._task_cfg["dynamics"][
            "hydrodynamics"
        ]["offset_lin_forward_damping_speed"]
        self.offset_nonlin_damping = self._task_cfg["dynamics"]["hydrodynamics"][
            "offset_nonlin_damping"
        ]
        self.scaling_damping = self._task_cfg["dynamics"]["hydrodynamics"][
            "scaling_damping"
        ]
        self.offset_added_mass = self._task_cfg["dynamics"]["hydrodynamics"][
            "offset_added_mass"
        ]
        self.scaling_added_mass = self._task_cfg["dynamics"]["hydrodynamics"][
            "scaling_added_mass"
        ]
        # Instantiate the task, reward and platform
        self.task = task_factory.get(task_cfg, reward_cfg, self._num_envs, self._device)
        self._penalties = parse_data_dict(Penalties(), penalty_cfg)
        self._num_observations = self.task._num_observations
        self._max_actions = 2  # Number of thrusters
        self._num_actions = 2  # Number of thrusters
        RLTask.__init__(self, name, env)
        # Instantiate the action and observations spaces
        self.set_action_and_observation_spaces()
        # Sets the initial positions of the target and platform
        self._fp_position = torch.tensor([0, 0.0, 0.5])
        self._default_marker_position = torch.tensor([0, 0, 1.0])
        self._marker = None
        
        # 泊位相关属性
        self._berths = []  # 存储所有泊位
        self._berth_enabled = True  # 是否启用泊位
        # Preallocate tensors
        self.actions = torch.zeros(
            (self._num_envs, self._max_actions),
            device=self._device,
            dtype=torch.float32,
        )
        self.heading = torch.zeros(
            (self._num_envs, 2), device=self._device, dtype=torch.float32
        )
        self.all_indices = torch.arange(
            self._num_envs, dtype=torch.int32, device=self._device
        )
        # Extra info
        self.extras = {}
        # Episode statistics
        self.episode_sums = self.task.create_stats({})
        self.add_stats(self._penalties.get_stats_name())
        self.add_stats(["normed_linear_vel", "normed_angular_vel", "actions_sum"])

        # obs variables
        self.root_pos = torch.zeros(
            (self._num_envs, 3), device=self._device, dtype=torch.float32
        )
        self.root_quats = torch.zeros(
            (self._num_envs, 4), device=self._device, dtype=torch.float32
        )
        self.root_velocities = torch.zeros(
            (self._num_envs, 6), device=self._device, dtype=torch.float32
        )
        self.euler_angles = torch.zeros(
            (self._num_envs, 3), device=self._device, dtype=torch.float32
        )

        # volume submerged
        self.high_submerged = torch.zeros(
            (self._num_envs), device=self._device, dtype=torch.float32
        )
        self.submerged_volume = torch.zeros(
            (self._num_envs), device=self._device, dtype=torch.float32
        )

        # forces to be applied
        self.hydrostatic_force = torch.zeros(
            (self._num_envs, 6), device=self._device, dtype=torch.float32
        )
        self.drag = torch.zeros(
            (self._num_envs, 6), device=self._device, dtype=torch.float32
        )
        self.thrusters = torch.zeros(
            (self._num_envs, 6), device=self._device, dtype=torch.float32
        )

        ##some tests for the thrusters

        self.stop = torch.tensor([0.0, 0.0], device=self._device)
        self.turn_right = torch.tensor([1.0, -1.0], device=self._device)
        self.turn_left = torch.tensor([-1.0, 1.0], device=self._device)
        self.forward = torch.tensor([1.0, 1.0], device=self._device)
        self.backward = -self.forward

        # 读取 YAML 开关，建议在 __init__ 末尾或 post_reset 开头就位
        env_cfg = getattr(self, "_task_cfg", {}).get("env", {}) or {}
        self.debug_berth = bool(env_cfg.get("debug_berth", False))
        self.debug_rewards = bool(env_cfg.get("debug_rewards", False))

        # === 新增：泊位朝向采样策略（YAML 以后再对齐）===
        br = env_cfg.get("berth", {}) or {}
        self.berth_yaw_mode     = str(br.get("yaw_mode", "episode"))   # "once"|"episode"|"success"|"seeded"|"fixed"
        self.berth_seed         = int(br.get("seed", 12345))
        self.berth_eval_lock    = bool(br.get("eval_lock", False))      # 评测时锁定，不再重采样
        self.berth_fixed_yaw_deg = br.get("fixed_yaw_deg", None)        # float 或 list[float]
        self.yaw_hold_k         = int(br.get("yaw_hold_k", 1))          # 泊位朝向保持K个回合
        self._berth_episode_count = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)
        
        # --- 泊位成功计数（用于success模式的重采样） ---
        self.berth_success_count = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)

        # --- 泊位奖励状态缓冲区 ---
        self.prev_d_gate = torch.zeros(self._num_envs, device=self._device)                 # 上一步门外距离 (N,)
        self.prev_d_center = torch.zeros(self._num_envs, device=self._device)               # 上一步门内中心距离 (N,)
        self.prev_outside = torch.ones(self._num_envs, dtype=torch.bool, device=self._device)  # 是否在门外
        self.gate_crossed_flags = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)  # 是否已给过穿门奖励
        self.mouth_hit_count = torch.zeros(self._num_envs, dtype=torch.int32, device=self._device)
        self.dwell_steps = torch.zeros(self._num_envs, device=self._device)                  # 门内停留步数
        self.steps_in_tolerance = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)  # 稳定计数

        return

    def _override_spawn_to_door_sector(self, env_ids: torch.Tensor, root_pos_w: torch.Tensor, root_rot_w: torch.Tensor):
        """把 get_spawns 返回的出生位姿，按"泊位门外扇区"覆写（仅覆写传入的 env_ids）。"""
        # 若泊位参数还没注入，直接返回原值
        need = (hasattr(self.task, "berth_size_LW") and hasattr(self.task, "berth_yaw")
                and hasattr(self.task, "berth_center_xy_env"))
        if not need or env_ids.numel() == 0:
            return root_pos_w, root_rot_w

        ids = env_ids.long()
        dev = self._device

        # 读取泊位几何（与 set_targets 一致）
        LW   = self.task.berth_size_LW[ids]           # [k,2] (L,W)
        L, W = LW[:,0], LW[:,1]
        halfL, halfW = 0.5*L, 0.5*W
        yaw  = self.task.berth_yaw[ids]               # [k] radians

        # 世界系泊位中心（= 目标点）：bc_w = bc_env + env_origin
        bc_env = self.task.berth_center_xy_env[ids, :2]           # [k,2] env
        env_o  = self._env_pos[ids, :2]                           # [k,2] world
        bc_w   = bc_env + env_o                                   # [k,2] world

        # 世界系基向量（与 set_targets 定义一致）
        # 开口方向 (+Y_B) 与 横向方向 (+X_B)
        open_dir_w    = torch.stack([-torch.sin(yaw),  torch.cos(yaw)], dim=-1)  # [k,2]
        lateral_dir_w = torch.stack([ torch.cos(yaw),  torch.sin(yaw)], dim=-1)  # [k,2]

        # 采样参数（也可用 getattr 从 cfg 读取）
        d_min = torch.full((ids.numel(),), self.min_spawn_dist, device=dev)
        d_max = torch.full((ids.numel(),), self.max_spawn_dist, device=dev)
        clearance     = torch.full((ids.numel(),), self.clearance, device=dev)
        lateral_span  = self.lateral_span
        heading_err   = math.radians(self.heading_err_deg)

        # 门外扇区采样：沿开口方向外推 + 横向带
        d_long = torch.rand_like(d_min) * (d_max - d_min) + d_min                # [k]
        u      = (torch.rand_like(halfW) * 2.0 - 1.0)
        x_B    = u * (halfW * lateral_span)                                     # [k]

        p_w = root_pos_w[ids].clone()
        p_w[:, 0] = bc_w[:,0] + (halfL + clearance + d_long) * open_dir_w[:,0] + x_B * lateral_dir_w[:,0]
        p_w[:, 1] = bc_w[:,1] + (halfL + clearance + d_long) * open_dir_w[:,1] + x_B * lateral_dir_w[:,1]
        # z 保持原来的高度
        p_w[:, 2] = root_pos_w[ids, 2]

        # 初始朝向：面向库内（-open_dir_w）±扰动
        theta_des = torch.atan2(-open_dir_w[:,1], -open_dir_w[:,0])
        yaw0      = theta_des + (torch.rand_like(theta_des)*2.0 - 1.0) * heading_err

        # 四元数顺序 [w,x,y,z]（只绕Z）
        cy = torch.cos(yaw0 * 0.5)
        sy = torch.sin(yaw0 * 0.5)
        quat = torch.zeros_like(root_rot_w[ids])
        quat[:, 0] = cy  # w
        quat[:, 3] = sy  # z

        # 回填覆写，仅更新这些 env_ids
        root_pos_w[ids] = p_w
        root_rot_w[ids] = quat
        return root_pos_w, root_rot_w

    def compute_berthing_reward(self, state, actions):
        """
        基于泊位坐标系的泊位奖励（矩形成功区版本）
        """
        import math
        import torch
        F = torch.nn.functional

        num_envs = self._num_envs
        device = self._device

        # 必要属性检查（来自 set_targets 注入）
        required = ["berth_center_xy_env", "R_env2berth", "berth_yaw", "berth_size_LW"]
        if any(not hasattr(self.task, k) for k in required):
            return torch.zeros(num_envs, device=device)

        # --- 1) 环境→泊位坐标 ---
        pos_env = state["position"][:, :2]
        vel_env = state.get("linear_velocity", torch.zeros((num_envs, 2), device=device))[:, :2]

        # 船体偏航（若你已有标准四元数->yaw函数，可替换这里）
        orient = state.get("orientation", None)
        if orient is not None:
            boat_yaw = torch.atan2(orient[:, 1], orient[:, 0])
        else:
            boat_yaw = torch.zeros(num_envs, device=device)

        bc_env = self.task.berth_center_xy_env                 # (N,2)
        R_e2b  = self.task.R_env2berth                         # (N,2,2)
        LW     = self.task.berth_size_LW                       # (N,2) -> [L, W]
        berth_yaw = self.task.berth_yaw                        # (N,)

        L, W = LW[:, 0], LW[:, 1]
        halfL, halfW = 0.5 * L, 0.5 * W

        rel_pos = pos_env - bc_env
        p_B = torch.einsum('nij,nj->ni', R_e2b, rel_pos)       # (N,2)
        v_B = torch.einsum('nij,nj->ni', R_e2b, vel_env)       # (N,2)

        xB, yB = p_B[:, 0], p_B[:, 1]                          # x: 横向, y: 纵向(+外)
        vxB, vyB = v_B[:, 0], v_B[:, 1]
        d_yaw = (boat_yaw - berth_yaw + math.pi) % (2*math.pi) - math.pi

        # --- 2) 读取配置（含矩形容差） ---
        env_cfg = self._task_cfg.get("env", {})
        tp = env_cfg.get("task_parameters", {})
        rp = env_cfg.get("reward_parameters", {})
        bw = env_cfg.get("berthing_reward_weights", {})

        # 任务参数（默认仍回落到 position_tolerance）
        pos_tol = float(tp.get("position_tolerance", 0.3))
        pos_tol_x = float(tp.get("position_tolerance_x", pos_tol))   # 矩形：横向容差
        pos_tol_y = float(tp.get("position_tolerance_y", pos_tol))   # 矩形：纵向容差
        hold_N = int(tp.get("kill_after_n_steps_in_tolerance", 50))
        yaw_tol = math.radians(float(tp.get("yaw_tolerance_deg", 10.0)))
        vel_tol = float(tp.get("velocity_tolerance", 0.1))
        kill_dist = float(tp.get("kill_dist", 70.0))

        # 奖励参数
        gate_bonus    = float(rp.get("gate_bonus", 5.0))
        dwell_bonus   = float(rp.get("dwell_bonus", 0.005))
        success_bonus = float(rp.get("success_bonus", 50.0))
        fail_penalty  = float(rp.get("fail_penalty", 20.0))

        # 权重
        soft_margin   = float(bw.get("soft_margin", 0.20))
        w_dist_out    = float(bw.get("distance_outside", 1.0))
        w_dist_in     = float(bw.get("distance_inside", 5.0))
        w_yaw         = float(bw.get("yaw_alignment", 1.5))
        w_lateral     = float(bw.get("lateral_deviation", 1.0))
        w_vel_in      = float(bw.get("velocity_inward", 0.5))
        w_vel_brake   = float(bw.get("velocity_brake", 2.0))
        w_barrier     = float(bw.get("barrier_penalty", 15.0))
        w_time        = float(bw.get("time_penalty", 0.01))

        # --- 3) 几何量 ---
        outside  = (yB > halfL)                                  # 门外判定
        d_gate   = (yB - halfL).clamp(min=0.0)                   # 外侧→门线距离 (N,)
        d_center = torch.sqrt(xB * xB + yB * yB + 1e-6)          # 到中心
        speed_B  = torch.sqrt(vxB * vxB + vyB * vyB + 1e-6)      # 速度模

        # --- 4) 距离/姿态/横向 shaping ---
        # 门外：有界化负项，远处不再恒负
        r_dist_out = -w_dist_out * torch.tanh(d_gate / 3.0)        # ∈[-w_out, 0]
        
        # 门内：改为"靠中心给正奖"，近中心≈+w_in，远处≈0
        r_center = w_dist_in * (1.0 - torch.tanh(d_center / 2.0))  # ∈[0, +w_in]
        
        # 汇总（门外只用 r_dist_out，门内用 r_center）
        r_dist = torch.where(outside, r_dist_out, r_center)

        # r_yaw = w_yaw * torch.cos(d_yaw)          # ←删掉
        r_yaw = torch.where(~outside, w_yaw * torch.cos(d_yaw),
                            torch.zeros_like(d_yaw))

        r_lat = torch.where(
            ~outside,
            -w_lateral * F.smooth_l1_loss(xB, torch.zeros_like(xB), reduction='none', beta=0.5),
            torch.zeros_like(xB)
        )

        # --- 5) 速度 shaping ---
        # 5.1 门外"进度差"奖励（只奖 d_gate 下降），并限流
        prog = (self.prev_d_gate - d_gate).clamp(min=0.0, max=0.5)
        r_vel_in = torch.where(outside, w_vel_in * prog, torch.zeros_like(prog))

        # --- 5.2 门内速度控制（近中心才逐步加强） ---
        # 距中心越近，越应该慢下来；R_brake 为刹车窗口半径
        R_brake = 0.6 * torch.minimum(halfL, halfW)   # 也可用常数，例如 1.5
        # 距离系数：远处≈0，近中心→1（用平方让过渡更柔）
        alpha = torch.clamp(1.0 - d_center / (R_brake + 1e-6), min=0.0, max=1.0) ** 2

        # 目标速度：越近越慢
        speed_target = torch.clamp(0.1 * d_center, max=0.3)

        # 只惩罚"超速超出 δ"的部分；δ 给一点缓冲，避免微小抖动也扣分
        delta = 0.10
        overspeed = torch.relu(speed_B - (speed_target + delta))

        # 刹车惩罚：只在门内生效，且随接近中心而放大
        r_vel_brake = torch.where(
            ~outside,
            -w_vel_brake * alpha * overspeed,
            torch.zeros_like(overspeed)
        )

        # 轻微的"慢速奖励"：低于目标速度时给一点点正反馈（不必太大）
        underspeed = torch.relu((speed_target - speed_B) - 0.05)
        r_vel_brake += torch.where(
            ~outside,
            0.2 * w_vel_brake * alpha * torch.tanh(underspeed),
            0.0
        )

        # --- 5.x 门内径向进度差（越靠中心越好） ---
        # 限流到每步最多 +0.5，避免爆
        w_prog_in = float(bw.get("center_progress", 1.0))   # 新权重（YAML/CLI可配）
        prog_center = (self.prev_d_center - d_center).clamp(min=0.0, max=0.5)
        r_prog_in = torch.where(~outside, w_prog_in * prog_center, torch.zeros_like(prog_center))

        # 5.3 精度项（门内且近中心且慢）
        r_precision = torch.where(
            (~outside) & (d_center < 1.0),
            2.0 * torch.exp(-d_center / 0.3) * torch.exp(-speed_B / 0.1),
            torch.zeros_like(d_center)
        )

        # --- 6) 碰墙惩罚（走廊限定 + 归一化二次罚） ---
        # 读取可选倍率（未配置则为 1.0）
        extra_penalty = float(self._task_cfg["env"].get("berthing_reward_weights", {}).get("barrier_extra", 1.0))

        inside_corridor = (~outside) & (yB >= -halfL)  # 进入口后、未到后墙之前

        # 侧墙：仅在走廊内启用；penetration 以 soft_margin 归一化，二次增长
        # 软墙几何基准：以内壁为基线（减去墙厚）
        tw = getattr(self.task, "berth_wall_t", torch.full_like(halfW, 0.10))
        inner = halfW - tw
        side_over = (torch.abs(xB) - (inner - soft_margin)).clamp(min=0.0)
        side_pen  = (side_over / (soft_margin + 1e-6)) ** 2
        side_pen  = torch.where(inside_corridor, side_pen, torch.zeros_like(side_pen))

        # 后墙：越过后墙 soft_margin 内才罚
        back_over = ((-halfL + soft_margin) - yB).clamp(min=0.0)
        back_pen  = (back_over / (soft_margin + 1e-6)) ** 2

        # 合成
        r_bar = -w_barrier * extra_penalty * (side_pen + back_pen)

        # 小护栏避免极端 penetration 带来的单步巨罚
        r_bar = torch.clamp(r_bar, min=-10.0)

        # 时间惩罚
        r_time = -w_time * torch.ones_like(xB)

        # --- 7) 事件奖励（一次性/递减） ---
        r_events = torch.zeros_like(xB)

        # —— 在函数一开始就做个快照（放在第一次用到 prev_outside 之前）——
        prev_out = getattr(self, 'prev_outside', torch.ones_like(outside))

        # 7.1 穿门只给一次（用旧的 prev_out 做穿门检测）
        gate_crossed    = prev_out & (~outside)
        new_gate_cross  = gate_crossed & (~self.gate_crossed_flags)
        r_events[new_gate_cross] += gate_bonus
        self.gate_crossed_flags |= gate_crossed

        # 7.2 门内停留递减
        self.dwell_steps = torch.where(~outside, self.dwell_steps + 1.0, torch.zeros_like(self.dwell_steps))
        dwell_reward = dwell_bonus * torch.exp(-self.dwell_steps / 100.0)
        r_events[~outside] += dwell_reward[~outside]

        # --- 8) 成功/失败（矩形成功区） ---
        pos_ok   = (~outside) & (torch.abs(xB) <= pos_tol_x) & (torch.abs(yB) <= pos_tol_y)
        yaw_ok   = (torch.abs(d_yaw) <= yaw_tol)
        vel_ok   = (speed_B <= vel_tol)
        ok       = pos_ok & yaw_ok & vel_ok

        # 稳定计数
        self.steps_in_tolerance = torch.where(ok, self.steps_in_tolerance + 1, torch.zeros_like(self.steps_in_tolerance))
        just_success = (self.steps_in_tolerance >= hold_N)

        # 失败：越界/离谱/远离
        # 软墙几何基准：以内壁为基线（减去墙厚）
        tw = getattr(self.task, "berth_wall_t", torch.full_like(halfW, 0.10))
        inner = halfW - tw
        guard = max(getattr(self, "as_entrance_guard", 0.05), 0.10)
        fail_side_inside = ((~outside) & (yB >= -halfL) & (torch.abs(xB) > (inner + 1e-3)))
        fail_side_mouth  = (outside & (yB <= (halfL + guard)) & (torch.abs(xB) > (inner + 1e-3)))
        fail_lateral = fail_side_inside
        fail_back    = (yB < (-(halfL + 0.3)))
        fail_far     = (d_center > kill_dist)
        fail_mask    = fail_lateral | fail_side_mouth | fail_back | fail_far

        # 终端奖励
        r_events[just_success] += success_bonus
        r_events[fail_mask]    -= fail_penalty
        
        # 成功计数+1（用于success模式的泊位重采样）
        if hasattr(self, "berth_success_count"):
            self.berth_success_count[just_success] += 1

        # 通知环境终止（用索引置位）
        if hasattr(self, "reset_buf"):
            mask = (just_success | fail_mask).view(-1)
            self.reset_buf[mask] = 1

        # --- 9) 汇总并裁剪 ---
        r_shaping = torch.clamp(
            r_dist + r_yaw + r_lat + r_vel_in + r_vel_brake + r_precision + r_prog_in + r_bar + r_time,
            -3.0, 5.0
        )

        # --- 10) 更新缓冲 ---
        self.prev_d_gate = torch.where(outside, d_gate, torch.zeros_like(d_gate))
        self.prev_d_center = torch.where(~outside, d_center, torch.zeros_like(d_center))   # ← 新增
        self.prev_outside = outside.clone()

        # ---- 记录到 TensorBoard（仅在 debug 模式）----
        if self._task_cfg["env"].get("debug_rewards", False):
            bm = self.extras.setdefault("berthing_metrics", {})
            # 注意要 .mean() 并在显卡上保持张量；rl-games会自己拿数值
            bm["r_center_mean"]     = r_center.mean()
            bm["r_prog_in_mean"]    = r_prog_in.mean()
            bm["r_vel_in_mean"]     = r_vel_in.mean()
            bm["r_vel_brake_mean"]  = r_vel_in.mean()

        return r_shaping + r_events

    def _extract_boat_yaw(self, state, v_B):
        """稳健获取船体偏航角：优先用 state['orientation']，否则用速度方向兜底。"""
        import torch
        orient = state.get("orientation", None)
        if orient is None:
            return torch.atan2(v_B[:, 0], v_B[:, 1])  # 用速度方向兜底（泊位系）
        # 若直接给 yaw（N,1）
        if orient.ndim == 2 and orient.shape[1] == 1:
            return orient[:, 0]
        # 若给的是 [cosψ, sinψ]（你当前实现就是这个）
        if orient.ndim == 2 and orient.shape[1] >= 2:
            return torch.atan2(orient[:, 1], orient[:, 0])
        # 其它情况，兜底
        return torch.atan2(v_B[:, 0], v_B[:, 1])

    def compute_adaptive_velocity_reward(self, state, xB, yB, vxB, vyB, d_center, outside, halfL, halfW, berth_yaw=None):
        """
        自适应速度奖励条目（远/中/近三段速 + 走廊横向速度惩罚 + 近场刹车 + 平滑度）
        返回: r_velocity ∈ [-5,2]，以及 target_speed 便于调参可视化。
        """
        import torch

        speed_B = torch.sqrt(vxB**2 + vyB**2 + 1e-6)

        # 先初始化
        r_velocity = torch.zeros_like(speed_B)

        # === 读取容差，定义中心盆地（跟 YAML 走，不再写死） ===
        tp = self._task_cfg.get("env", {}).get("task_parameters", {})
        pos_tol_x = float(tp.get("position_tolerance_x", tp.get("position_tolerance", 0.5)))
        pos_tol_y = float(tp.get("position_tolerance_y", tp.get("position_tolerance", 0.5)))
        stop_margin_x = pos_tol_x + 0.15
        stop_margin_y = pos_tol_y + 0.15
        in_stop_basin = (~outside) & (torch.abs(xB) < stop_margin_x) & (torch.abs(yB) < stop_margin_y)

        # === 入口/走廊前段 提速 + 对齐 ===
        # 1) 走廊内且离中心还远：太慢要扣（避免门口龟速）
        mask_far_in_corridor = (~outside) & ( (xB**2 + yB**2) > (1.2**2) )
        v_floor = 0.35  # 你可以 0.30~0.45 之间试
        slow_mask = mask_far_in_corridor & (speed_B < v_floor)
        r_velocity += torch.where(slow_mask, -0.6 * (v_floor - speed_B), torch.zeros_like(speed_B))

        # 2) 入库对齐：在走廊内，角度偏差越大扣越多（越靠近侧墙权重越大）
        # 获取船体与泊位 +Y_B 的夹角
        if berth_yaw is not None:
            boat_yaw = self._extract_boat_yaw(state, torch.stack([vxB, vyB], dim=-1))
            d_yaw = (boat_yaw - berth_yaw + math.pi) % (2*math.pi) - math.pi
            w_align = torch.clamp(torch.abs(xB) / (halfW + 1e-6), 0.0, 1.0)   # 贴墙时加大对齐权
            r_velocity += -1.2 * w_align * torch.abs(d_yaw)                   # 角度单位是 rad

        # 3) 区域划分
        far_field = (d_center > 10.0) | outside
        mid_field = (~outside) & (d_center > 3.0) & (d_center <= 10.0)
        near_field = (~outside) & (d_center <= 3.0)

        # 2) 动态目标速度
        target_speed = torch.zeros_like(speed_B)
        target_speed[far_field] = torch.clamp(0.15 * d_center[far_field], min=0.3, max=1.0)
        target_speed[mid_field] = 0.3 + 0.4 * (d_center[mid_field] - 3.0) / 7.0   # 0.3~0.7
        target_speed[near_field] = torch.clamp(0.05 * d_center[near_field], min=0.02, max=0.15)
        
        # 盆地中心约束
        target_speed = torch.where(in_stop_basin, torch.zeros_like(target_speed), target_speed)

        # 3) 方向一致性（朝中心）
        vel_mag = speed_B.clamp(min=1e-6)
        tgt_x = -xB / (d_center + 1e-6)
        tgt_y = -yB / (d_center + 1e-6)
        vx_u = vxB / vel_mag
        vy_u = vyB / vel_mag
        direction_alignment = vx_u * tgt_x + vy_u * tgt_y  # [-1,1]

        # 4) 大小奖励：不对称惩罚（超速重、欠速轻）
        speed_diff = speed_B - target_speed
        speed_penalty = torch.where(speed_diff > 0, -2.0 * speed_diff**2, -0.5 * speed_diff**2)

        # 5) 合成（近场更看重方向）
        r_velocity = torch.where(
            near_field,
            0.3 * torch.exp(speed_penalty) + 0.7 * direction_alignment,
            0.5 * torch.exp(speed_penalty) + 0.5 * direction_alignment
        )

        # 6) 走廊横向速度惩罚（门内且 y 在走廊范围）
        corridor_mask = (~outside) & (yB > -halfL) & (yB < halfL)
        r_velocity += torch.where(corridor_mask, -0.5 * torch.abs(vxB), torch.zeros_like(vxB))

        # 7) 近成功区刹车（接近成功容差但速度过快要额外扣）
        near_success = (torch.abs(xB) < 0.5) & (torch.abs(yB) < 0.5) & (~outside)
        r_velocity += torch.where(near_success & (speed_B > 0.1),
                                  -5.0 * (speed_B - 0.1),
                                  torch.zeros_like(speed_B))

        # 7.5) 后墙安全刹车：越靠近后墙，向后速度（-vyB）扣得越重
        back_margin = 0.5  # 由 0.35 改为 0.5
        proximity = torch.clamp((-halfL + back_margin - yB) / back_margin, 0.0, 1.0)
        towards_back_speed = torch.relu(-vyB)  # 只罚朝后墙速度
        r_velocity += -6.0 * proximity * towards_back_speed  # 由 -4.0 改为 -6.0

        # 8) 中心盆地强粘性（只在接近成功时压速度；别处不动）
        # in_stop_basin 已在函数前半段按容差(+0.25)定义好了
        r_velocity += torch.where(
            in_stop_basin,
            -5.0 * torch.abs(vyB) - 3.0 * torch.abs(vxB),   # y 向更重
            torch.zeros_like(vyB)
        )

        # 9) 平滑度（避免突然大幅减速）
        if hasattr(self, 'prev_speed'):
            speed_change = torch.abs(speed_B - self.prev_speed)
            r_velocity += -torch.clamp(speed_change - 0.1, min=0.0) * 2.0
        self.prev_speed = speed_B.clone()

        return torch.clamp(r_velocity, min=-5.0, max=2.0), target_speed

    def set_action_and_observation_spaces(self) -> None:
        """
        Sets the action and observation spaces."""

        # Defines the observation space
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(
                    np.ones(self._num_observations) * -np.Inf,
                    np.ones(self._num_observations) * np.Inf,
                ),
            }
        )

        # Defines the action space
        if self._discrete_actions == "MultiDiscrete":
            # RLGames implementation of MultiDiscrete action space requires a tuple of Discrete spaces
            self.action_space = spaces.Tuple([spaces.Discrete(2)] * self._max_actions)
        elif self._discrete_actions == "Continuous":
            self.action_space = spaces.Box(
                low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32
            )
        elif self._discrete_actions == "Discrete":
            raise NotImplementedError("The Discrete control mode is not supported.")
        else:
            raise NotImplementedError(
                "The requested discrete action type is not supported."
            )

    def add_stats(self, names: List[str]) -> None:
        """
        Adds training statistics to be recorded during training.

        Args:
            names (List[str]): list of names of the statistics to be recorded."""

        for name in names:
            torch_zeros = lambda: torch.zeros(
                self._num_envs,
                dtype=torch.float,
                device=self._device,
                requires_grad=False,
            )
            if not name in self.episode_sums.keys():
                self.episode_sums[name] = torch_zeros()

    def cleanup(self) -> None:
        """
        Prepares torch buffers for RL data collection."""

        # prepare tensors
        self.obs_buf = {
            "state": torch.zeros(
                (self._num_envs, self._num_observations),
                device=self._device,
                dtype=torch.float,
            ),
        }

        self.states_buf = torch.zeros(
            (self._num_envs, self.num_states), device=self._device, dtype=torch.float
        )
        self.rew_buf = torch.zeros(
            self._num_envs, device=self._device, dtype=torch.float
        )
        self.reset_buf = torch.ones(
            self._num_envs, device=self._device, dtype=torch.long
        )
        self.progress_buf = torch.zeros(
            self._num_envs, device=self._device, dtype=torch.long
        )
        self.extras = {}

    def set_up_scene(self, scene) -> None:
        """
        Sets up the USD scene inside Omniverse for the task.

        Args:
            scene (Usd.Stage): the USD scene to be set up."""

        # Add the floating platform, and the marker
        self.get_heron()
        self.get_target()
        self.get_USV_dynamics()

        RLTask.set_up_scene(self, scene, replicate_physics=False)

        # Collects the interactive elements in the scene
        root_path = "/World/envs/.*/heron"
        self._heron = HeronView(prim_paths_expr=root_path, name="heron_view")

        # Add views to scene
        scene.add(self._heron)
        scene.add(self._heron.base)

        scene.add(self._heron.thruster_left)
        scene.add(self._heron.thruster_right)

        # Add arrows to scene if task is go to pose
        scene, self._marker = self.task.add_visual_marker_to_scene(scene)
        
        # 创建泊位
        if self._berth_enabled:
            self.create_berths()
        
        return

    def get_heron(self):
        """
        Adds the floating platform to the scene."""

        fp = Heron(
            prim_path=self.default_zero_env_path + "/heron",
            name="heron",
            translation=self._fp_position,
            # cfg=self._heron_cfg,
        )
        self._sim_config.apply_articulation_settings(
            "heron",
            get_prim_at_path(fp.prim_path),
            self._sim_config.parse_actor_config("heron"),
        )

    def get_target(self) -> None:
        """
        Adds the visualization target to the scene."""

        self.task.generate_target(
            self.default_zero_env_path, self._default_marker_position
        )

    def get_USV_dynamics(self):
        """create physics"""
        self.hydrostatics = HydrostaticsObject(
            num_envs=self.num_envs,
            device=self._device,
            water_density=self.water_density,
            gravity=self.gravity,
            metacentric_width=self.box_width / 2,
            metacentric_length=self.box_length / 2,
            average_hydrostatics_force_value=self.average_hydrostatics_force_value,
            amplify_torque=self.amplify_torque,
            offset_added_mass=self.offset_added_mass,
            scaling_added_mass=self.scaling_added_mass,
            alpha=self.alpha,
            last_time=self.last_time,
        )
        self.hydrodynamics = HydrodynamicsObject(
            task_cfg=self._task_cfg["env"]["disturbances"]["drag"],
            num_envs=self.num_envs,
            device=self._device,
            water_density=self.water_density,
            gravity=self.gravity,
            linear_damping=self.linear_damping,
            quadratic_damping=self.quadratic_damping,
            linear_damping_forward_speed=self.linear_damping_forward_speed,
            offset_linear_damping=self.offset_linear_damping,
            offset_lin_forward_damping_speed=self.offset_lin_forward_damping_speed,
            offset_nonlin_damping=self.offset_nonlin_damping,
            scaling_damping=self.scaling_damping,
            offset_added_mass=self.offset_added_mass,
            scaling_added_mass=self.scaling_added_mass,
            alpha=self.alpha,
            last_time=self.last_time,
        )
        self.thrusters_dynamics = DynamicsFirstOrder(
            task_cfg=self._task_cfg["env"]["disturbances"]["thruster"],
            num_envs=self.num_envs,
            device=self._device,
            timeConstant=self.timeConstant,
            dt=self.dt,
            numberOfPointsForInterpolation=self.numberOfPointsForInterpolation,
            interpolationPointsFromRealDataLeft=self.interpolationPointsFromRealDataLeft,
            interpolationPointsFromRealDataRight=self.interpolationPointsFromRealDataRight,
            coeff_neg_commands=self.neg_cmd_coeff,
            coeff_pos_commands=self.pos_cmd_coeff,
            cmd_lower_range=self.cmd_lower_range,
            cmd_upper_range=self.cmd_upper_range,
        )

    def create_berths(self) -> None:
        """
        为每个环境创建泊位
        """
        # 获取目标点位置（红球位置）
        target_pos = self._default_marker_position
        
        # 为每个环境创建泊位
        for env_id in range(self._num_envs):
            # 计算环境特定的目标位置
            env_target_pos = target_pos.clone()
            
            # 创建泊位路径
            berth_path = f"/World/envs/env_{env_id}/berth"
            
            # 创建泊位实例
            berth = Berth(
                prim_path=berth_path,
                target_position=env_target_pos,
                boat_length=self.box_length,
                boat_width=self.box_width,
                wall_thickness=0.10,
                wall_height=self.heron_zero_height * 0.5,  # 船高度的一半
                z_base=0.0,
                alpha=0.3
            )
            
            self._berths.append(berth)

    def update_state(self) -> None:
        """
        Updates the state of the system."""

        # Collects the position and orientation of the platform
        self.root_pos, self.root_quats = self._heron.get_world_poses(clone=True)
        # Remove the offset from the different environments
        root_positions = self.root_pos - self._env_pos
        # Collects the velocity of the platform
        self.root_velocities = self._heron.get_velocities(clone=True)
        root_velocities = self.root_velocities.clone()
        # Cast quaternion to Yaw
        siny_cosp = 2 * (
            self.root_quats[:, 0] * self.root_quats[:, 3]
            + self.root_quats[:, 1] * self.root_quats[:, 2]
        )
        cosy_cosp = 1 - 2 * (
            self.root_quats[:, 2] * self.root_quats[:, 2]
            + self.root_quats[:, 3] * self.root_quats[:, 3]
        )
        orient_z = torch.atan2(siny_cosp, cosy_cosp)
        # Add noise on obs
        root_positions = self.ON.add_noise_on_pos(root_positions)
        root_velocities = self.ON.add_noise_on_vel(root_velocities)
        orient_z = self.ON.add_noise_on_heading(orient_z)
        # Compute the heading
        self.heading[:, 0] = torch.cos(orient_z)
        self.heading[:, 1] = torch.sin(orient_z)

        # get euler angles
        self.get_euler_angles(self.root_quats)  # rpy roll pitch yaws

        # body underwater
        self.high_submerged[:] = torch.clamp(
            (self.heron_zero_height) - self.root_pos[:, 2],
            0,
            self.heron_zero_height + 20,  # TODO: Hardcoded value
        )
        self.submerged_volume[:] = torch.clamp(
            self.high_submerged * self.waterplane_area, 0, self.max_volume
        )
        self.box_is_under_water = torch.where(
            self.high_submerged[:] > 0, 1.0, 0.0
        ).unsqueeze(0)

        # Dump to state
        self.current_state = {
            "position": root_positions[:, :2],
            "orientation": self.heading,
            "linear_velocity": root_velocities[:, :2],
            "angular_velocity": root_velocities[:, -1],
        }

    def get_euler_angles(self, quaternions):
        """quaternions to euler"""

        w, x, y, z = quaternions.unbind(dim=1)
        rotation_matrices = torch.stack(
            [
                1 - 2 * y**2 - 2 * z**2,
                2 * x * y - 2 * w * z,
                2 * x * z + 2 * w * y,
                2 * x * y + 2 * w * z,
                1 - 2 * x**2 - 2 * z**2,
                2 * y * z - 2 * w * x,
                2 * x * z - 2 * w * y,
                2 * y * z + 2 * w * x,
                1 - 2 * x**2 - 2 * y**2,
            ],
            dim=1,
        ).view(-1, 3, 3)

        angle_x = torch.atan2(rotation_matrices[:, 2, 1], rotation_matrices[:, 2, 2])
        angle_y = torch.asin(-rotation_matrices[:, 2, 0])
        angle_z = torch.atan2(rotation_matrices[:, 1, 0], rotation_matrices[:, 0, 0])

        euler = torch.stack((angle_x, angle_y, angle_z), dim=1)

        """quaternions to euler"""
        self.euler_angles[:, :] = euler

    def get_observations(self) -> Dict[str, torch.Tensor]:
        """
        Gets the observations of the task to be passed to the policy.

        Returns:
            observations: a dictionary containing the observations of the task."""

        # implement logic to retrieve observation states
        self.update_state()
        # Get the state
        self.obs_buf["state"] = self.task.get_state_observations(
            self.current_state, self._observation_frame
        )

        observations = {self._heron.name: {"obs_buf": self.obs_buf}}

        # 几何一致性自检（只在reset后首帧）
        if getattr(self, "debug_berth", False) and self.progress_buf.max() == 0:
            self._debug_after_obs(self.current_state)

        return observations

    def pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        This function implements the logic to be performed before physics steps.

        Args:
            actions (torch.Tensor): the actions to be applied to the platform."""

        # If is not playing skip
        if not self._env._world.is_playing():
            return
        # Check which environment need to be reset
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        # Reset the environments (Robots)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)
        # Collect actions
        actions = actions.clone().to(self._device)
        self.actions = actions

        # Debug : Set actions
        # self.actions = torch.ones_like(self.actions) * 0.0

        # Remap actions to the correct values
        if self._discrete_actions == "MultiDiscrete":
            # If actions are multidiscrete [0, 1]
            thrust_cmds = self.actions.float() * 2 - 1
        elif self._discrete_actions == "Continuous":
            # Transform continuous actions to [-1, 1] discrete actions.
            thrust_cmds = self.actions.float()
        else:
            raise NotImplementedError("")

        # Applies the thrust multiplier
        thrusts = thrust_cmds

        # Adds random noise on the actions
        thrusts = self.AN.add_noise_on_act(thrusts)

        # --- Center basin smooth throttle (SAFE + anti-back-kick) ---
        need = all(hasattr(self.task, k) for k in ["berth_center_xy_env","R_env2berth"]) \
               and hasattr(self, 'root_pos')

        if need:
            bc_env = self.task.berth_center_xy_env
            R_e2b  = self.task.R_env2berth
            # 使用 root_pos 而不是 current_state，因为 pre_physics_step 在 update_state 之前被调用
            pos_env = self.root_pos[:, :2] - self._env_pos[:, :2]  # 转换为环境坐标系
            p_B = torch.einsum('nij,nj->ni', R_e2b, pos_env - bc_env)
            xB, yB = p_B[:, 0], p_B[:, 1]

            # 同步计算泊位系速度（供侧墙刹车/限速/对齐使用）
            vel_env = self.root_velocities[:, :2]
            v_B = torch.einsum('nij,nj->ni', R_e2b, vel_env)
            vxB, vyB = v_B[:, 0], v_B[:, 1]

            tp = self._task_cfg.get("env", {}).get("task_parameters", {})
            tol_x = float(tp.get("position_tolerance_x", tp.get("position_tolerance", 0.3)))
            tol_y = float(tp.get("position_tolerance_y", tp.get("position_tolerance", 0.3)))
            sx, sy = tol_x + 0.15, tol_y + 0.15

            nx = (torch.abs(xB) / (sx + 1e-6)).clamp(0, 1)
            ny = (torch.abs(yB) / (sy + 1e-6)).clamp(0, 1)
            t  = torch.maximum(nx, ny)
            s  = t * t * (3 - 2 * t)  # smoothstep

            # 近中心最小档（更保守）：0.15×；出盆地回到 1.0×
            throttle_scale = 0.15 + 0.85 * s
            thrusts = thrusts * throttle_scale.unsqueeze(-1)

            # 仅在"中心盆地 & 后半区(y<0)"禁止倒车（避免最后一步倒冲）
            in_basin = (torch.abs(xB) < sx) & (torch.abs(yB) < sy)
            back_half = (yB < 0)
            mask = in_basin & back_half
            thrusts[mask] = torch.clamp(thrusts[mask], min=0.0)

            # ---- Side-wall anti-scrape yaw damper ----
            # 只在走廊内做侧墙阻尼（避免在库外/开口处干扰）
            halfW = 0.5 * self.task.berth_size_LW[:, 1]  # 泊位半宽
            halfL = 0.5 * self.task.berth_size_LW[:, 0]  # 泊位半长
            in_corridor = ( (yB > -halfL) & (yB < (halfL - self.as_entrance_guard)) )

            side_margin = 0.35
            tw = getattr(self.task, "berth_wall_t", torch.full_like(halfW, 0.10))
            inner = halfW - tw
            boat_half_w = 0.5 * float(self.box_width)
            prox_side = ((torch.abs(xB) + boat_half_w) - (inner - side_margin)) / (side_margin + 1e-6)
            prox_side = torch.clamp(prox_side, 0.0, 1.0)

            # 提取"转向差分"（假设 thrusts=[left,right]，差分>0 表示"向右转"）
            diff = thrusts[:, 0] - thrusts[:, 1]
            mean = 0.5 * (thrusts[:, 0] + thrusts[:, 1])

            near_right = xB > (inner - 0.5 * side_margin)
            near_left  = xB < -(inner - 0.5 * side_margin)

            # 只在"真的要往贴墙方向转"时触发
            mask_right = near_right & (diff > 0) & in_corridor
            mask_left  = near_left  & (diff < 0) & in_corridor

           
            alpha_max = 0.4
            damp = 1.0 - alpha_max * prox_side  # broadcast ok

            # 应用阻尼（只改对应 mask 的 diff）
            diff_adj = diff.clone()
            diff_adj[mask_right] = diff[mask_right] * damp[mask_right]
            diff_adj[mask_left]  = diff_adj[mask_left] * damp[mask_left]

            # 还原到左右推力
            thrusts[:, 0] = mean + 0.5 * diff_adj
            thrusts[:, 1] = mean - 0.5 * diff_adj

            # --- Side-wall safety bubble (danger-based action gating) ---
            # 读取泊位尺寸
            LW = self.task.berth_size_LW
            halfW = 0.5 * LW[:, 1]
            halfL = 0.5 * LW[:, 0]

            bw = self._task_cfg.get("env", {}).get("berthing_reward_weights", {})
            danger_margin = float(bw.get("danger_margin", 0.20))

            # 与奖励一致的危险度定义（只在走廊内考虑）
            in_corridor = ( (yB > -halfL) & (yB < (halfL - self.as_entrance_guard)) )
            # 软墙几何基准：以内壁为基线（减去墙厚）
            tw = getattr(self.task, "berth_wall_t", torch.full_like(halfW, 0.10))
            inner = halfW - tw
            d_side = (inner - torch.abs(xB)).clamp(min=0.0)
            danger = torch.zeros_like(xB)
            danger[in_corridor] = torch.clamp(
                (danger_margin - d_side[in_corridor]) / (danger_margin + 1e-6), 0.0, 1.0
            )

            # 参数：靠墙越近，越收油门/差动；再加一个轻微的离墙偏置
            g_throttle = self.as_g_throttle
            g_diff     = self.as_g_diff
            g_bias     = self.as_g_bias

            # 分解为中值/半差（便于只压制差动）
            mid  = 0.5 * (thrusts[:, 0] + thrusts[:, 1])
            hdf  = 0.5 * (thrusts[:, 0] - thrusts[:, 1])

            # 危险度退火（前几千步更温和）
            anneal = torch.clamp(torch.tensor(self.step / 2000.0, device=self._device), 0.0, 1.0)

            # --- Heading error gate for turn allowance (compute d_yaw early for gating) ---
            # Boat yaw from root quaternions
            w, x, y, z = self.root_quats.unbind(dim=1)
            num = 2.0 * (w * z + x * y)
            den = 1.0 - 2.0 * (y * y + z * z)
            boat_yaw_gate = torch.atan2(num, den)

            # Desired heading using ILOS + crab angle estimated from boat velocity (berth frame)
            psi_path_gate = self.task.berth_yaw - math.pi * 0.5
            if (self.kp_ilos != 0.0) or (self.ki_ilos != 0.0):
                arg_gate = - self.kp_ilos * xB - self.ki_ilos * self.ilos_zeta
                chi_d_gate = psi_path_gate + torch.atan(arg_gate)
            else:
                chi_d_gate = psi_path_gate + torch.atan(- self.kp_ilos * xB)

            # Use boat velocity to estimate crab angle for gating
            if hasattr(self, "root_velocities"):
                vel_env_gate = self.root_velocities[:, :2]
                v_B_gate = torch.einsum('nij,nj->ni', R_e2b, vel_env_gate)
                vxB_gate, vyB_gate = v_B_gate[:, 0], v_B_gate[:, 1]
            else:
                vxB_gate = torch.zeros_like(xB)
                vyB_gate = torch.ones_like(xB) * (-0.3)
            beta_gate = torch.atan2(vxB_gate, (-vyB_gate).clamp(min=1e-3))
            k_beta_gate = float(getattr(self, "as_k_beta", getattr(self, "as_current_heading_gain", 0.8)))
            desired_yaw_gate = chi_d_gate + k_beta_gate * beta_gate
            d_yaw = (boat_yaw_gate - desired_yaw_gate + math.pi) % (2.0 * math.pi) - math.pi
            yaw_gate = (1.0 - (torch.abs(d_yaw) / 0.5)).clamp(0.0, 1.0)

            # 1) 降油门：越危险越收
            throttle_scale2 = 1.0 - g_throttle * (danger ** 2) * anneal
            # 保底推力，防止门控把油门吃光导致“卡死”
            throttle_scale2 = torch.clamp(throttle_scale2, 0.35, 1.0)

            # 2) 降差动：越危险越少转
            hdf = hdf * (1.0 - g_diff * (danger ** 2) * anneal * yaw_gate)

            # 3) 轻微离墙偏置：+X_B 说明靠右墙（xB>0），给负的差动把船往左拽一点
            hdf = hdf - g_bias * torch.sign(xB) * danger * anneal
            # 3bis) Lateral-velocity anti-drift: pull back toward corridor center under side-current
            if hasattr(self, "root_velocities"):
                vel_env = self.root_velocities[:, :2]
                v_B = torch.einsum('nij,nj->ni', R_e2b, vel_env)
                vxB = v_B[:, 0]
            else:
                vxB = torch.zeros_like(xB)
            hdf = hdf - self.as_k_vx * torch.sign(xB) * torch.abs(vxB) * danger

            # === Mouth-outer safety bubble (outside & near the mouth) ===
            entrance_guard = float(getattr(self, "as_entrance_guard", 0.10))
            mouth_outer = (yB > halfL) & (yB < (halfL + entrance_guard + 0.15))

            # 与侧墙的口外邻近度 
            tw = getattr(self.task, "berth_wall_t", torch.full_like(halfW, 0.10))
            inner = (halfW - tw).clamp(min=0.05)
            boat_half_w = 0.5 * float(self.box_width)
            m_margin = 0.25
            prox_mouth = ((torch.abs(xB) + boat_half_w) - (inner - m_margin)) / (m_margin + 1e-6)
            prox_mouth = torch.clamp(prox_mouth, 0.0, 1.0)

            # 若正在"向墙推进"，强度更高
            push_into_wall = (xB * vxB) > 0.0
            scale = (0.4 + 0.4 * push_into_wall.float()) * prox_mouth  # 0.4~0.8

            # 轻量抑制：口外只用走廊内强度的 60%
            throttle_scale2 = torch.where(
                mouth_outer, throttle_scale2 * (1.0 - self.as_g_throttle * 0.6 * scale), throttle_scale2
            )
            # 保底推力，防止门控把油门吃光导致“卡死”
            throttle_scale2 = torch.clamp(throttle_scale2, 0.35, 1.0)
            # yaw_gate：朝向误差门控（误差越大，越少抑制差动，以便“能转进门”）
            yaw_gate = (1.0 - (torch.abs(d_yaw) / 0.5)).clamp(0.0, 1.0)
            hdf = torch.where(
                mouth_outer, hdf * (1.0 - self.as_g_diff * 0.6 * scale * yaw_gate), hdf
            )
            # 离墙小偏置（根据 xB 把船轻推向中线）
            hdf = torch.where(
                mouth_outer, hdf - (self.as_g_bias * 0.6) * torch.sign(xB) * scale, hdf
            )
            # 按 |vxB| 抑制：横飘越大，抑制越强
            hdf = torch.where(
                mouth_outer, hdf - self.as_k_vx * 0.6 * torch.sign(xB) * torch.abs(vxB) * scale, hdf
            )

            # 4) 门内止转阀：进入中心盆地且速度很低时，强行取消差动，防"原地打转"
            sx, sy = tol_x, tol_y  
            in_basin = (torch.abs(xB) < sx) & (torch.abs(yB) < sy)
            # 近似速度门限，直接用中值推力代替（简单有效）
            low_throttle = (mid.abs() < 0.2)
            
            # ---- Basin spin gate: only stop turning when heading is already aligned ----
            # 从当前 root_quats 计算船体 yaw
            w, x, y, z = self.root_quats.unbind(dim=1)
            num = 2.0 * (w * z + x * y)
            den = 1.0 - 2.0 * (y * y + z * z)
            boat_yaw = torch.atan2(num, den)

            # Desired heading with LOS/ILOS + crab angle (current compensation)
            psi_path = self.task.berth_yaw - math.pi * 0.5  # path tangent towards berth (-Y_B)

            # ILOS: zeta integrates cross-track error xB
            if (self.kp_ilos != 0.0) or (self.ki_ilos != 0.0):
                self.ilos_zeta = torch.clamp(self.ilos_zeta + self.dt * xB, -0.5, 0.5)
                arg = - self.kp_ilos * xB - self.ki_ilos * self.ilos_zeta
                chi_d = psi_path + torch.atan(arg)
            else:
                chi_d = psi_path + torch.atan(- self.kp_ilos * xB)  # kp_ilos=0 -> psi_path

            # Crab angle from side-current in berth frame
            # Convert flow_vel to proper tensor format for batch processing
            flow_env = torch.tensor(self.flow_vel, device=self._device, dtype=torch.float32)[:2]  # [2]
            flow_env_batch = flow_env.unsqueeze(0).expand(self._num_envs, -1)  # [N, 2]
            flow_B   = torch.einsum('nij,nj->ni', R_e2b, flow_env_batch)  # [N,2]
            # Crab speed reference from actual forward speed in berth frame (vyB)
            if hasattr(self, "root_velocities"):
                vel_env = self.root_velocities[:, :2]
                v_B     = torch.einsum('nij,nj->ni', R_e2b, vel_env)
                vyB     = v_B[:, 1].abs()
                U_ref   = torch.clamp(vyB, min=0.15, max=0.8)
            else:
                U_ref   = torch.full((self._num_envs,), 0.30, device=self._device, dtype=torch.float32)
            beta     = torch.atan2(flow_B[:, 0], U_ref + 1e-3) * self.as_crab_gain

            desired_yaw = chi_d + beta
            d_yaw = (boat_yaw - desired_yaw + math.pi) % (2.0 * math.pi) - math.pi

            # Light yaw damper outside the berth to prevent drift/spiral
            LW     = self.task.berth_size_LW
            halfL  = 0.5 * LW[:, 0]
            outside = (yB > halfL)
            hdf = hdf - self.as_outer_yaw * d_yaw * outside.float()

            # 与 YAML 同步的容差（12°），对齐判据稍紧一点（60%*容差）
            yaw_tol = math.radians(self._task_cfg["env"]["task_parameters"].get("yaw_tolerance_deg", 12.0))
            aligned = (torch.abs(d_yaw) <= 0.6 * yaw_tol)

            # 只有在"盆地 & 低油门 & 已对齐"时，才把差动清零；否则允许转动去对齐
            stop_spin = in_basin & low_throttle & aligned
            hdf = torch.where(stop_spin, torch.zeros_like(hdf), hdf)
            
            # ---- Basin alignment boost (optional steady-state enhancement) ----
            # 在"盆地 & 低油门 & 未对齐"时，给一个极小的差动朝着对齐方向，避免策略在噪声里犹豫
            need_align = in_basin & low_throttle & (~aligned)
            # 将角度误差缩放到 [-0.3, 0.3] 的小差动
            align_cmd = torch.clamp(d_yaw / math.radians(30.0), -1.0, 1.0) * 0.3
            hdf = torch.where(need_align, align_cmd, hdf)

            # --- Back-wall brake gate: forbid net backward thrust near the back wall ---
            back_guard = (yB < (-halfL + self.as_back_buf)) & ((0.5 * (thrusts[:, 0] + thrusts[:, 1])) < 0.0)

            mid = 0.5 * (thrusts[:, 0] + thrusts[:, 1])
            hdf = 0.5 * (thrusts[:, 0] - thrusts[:, 1])

            # 近后墙时：保证正向推（mid>=back_push），并把差动清零（不允许原地打转）
            mid = torch.where(
                back_guard,
                torch.maximum(mid, torch.full_like(mid, self.as_back_push)),
                mid,
            )
            hdf = torch.where(back_guard, torch.zeros_like(hdf), hdf)

            # ——保留 mid/hdf 供后续统一合成——

            # 复合回左右推力，并整体再乘降油门
            left  = (mid + hdf) * throttle_scale2
            right = (mid - hdf) * throttle_scale2
            thrusts = torch.stack([left, right], dim=-1)
        else:
            # 张量未就绪时保守一帧，防止"未限幅的全功率脉冲"
            thrusts = 0.15 * thrusts

        # Clip the actions
        thrusts = torch.clamp(thrusts, -1.0, 1.0)

        # clear actions for reset envs
        thrusts[reset_env_ids] = 0

        self.thrusters_dynamics.set_target_force(thrusts)

        return

    def apply_forces(self) -> None:
        """
        Applies all the forces to the platform and its thrusters."""

        disturbance_forces = self.UF.get_disturbance_forces(self.root_pos)
        torque_disturbance = self.TD.get_torque_disturbance(self.root_pos)

        # Hydrostatic force
        self.hydrostatic_force[:, :] = (
            self.hydrostatics.compute_archimedes_metacentric_local(
                self.submerged_volume, self.euler_angles, self.root_quats
            )
        )
        # Hydrodynamic forces
        self.drag[:, :] = self.hydrodynamics.ComputeHydrodynamicsEffects(
            0.01,
            self.root_quats,
            self.root_velocities[:, :],
            self.use_water_current,
            self.flow_vel,
        )  

        self.thrusters[:, :] = self.thrusters_dynamics.update_forces()

        self._heron.base.apply_forces_and_torques_at_pos(
            forces=disturbance_forces
            + self.hydrostatic_force[:, :3]
            + self.drag[:, :3],
            torques=torque_disturbance
            + self.hydrostatic_force[:, 3:]
            + self.drag[:, 3:],
            is_global=False,
        )

        self._heron.thruster_left.apply_forces_and_torques_at_pos(
            forces=self.thrusters[:, :3], is_global=False
        )
        self._heron.thruster_right.apply_forces_and_torques_at_pos(
            forces=self.thrusters[:, 3:], is_global=False
        )

    def post_reset(self):
        """
        This function implements the logic to be performed after a reset.
        """

        # implement any logic required for simulation on-start here
        self.root_pos, self.root_rot = self._heron.get_world_poses()
        self.root_velocities = self._heron.get_velocities()
        self.dof_pos = self._heron.get_joint_positions()
        self.dof_vel = self._heron.get_joint_velocities()

        self.initial_root_pos, self.initial_root_rot = (
            self.root_pos.clone(),
            self.root_rot.clone(),
        )
        self.initial_pin_pos = self._env_pos
        self.initial_pin_rot = torch.zeros(
            (self.num_envs, 4), dtype=torch.float32, device=self._device
        )
        self.initial_pin_rot[:, 0] = 1

        # control parameters
        self.thrusts = torch.zeros(
            (self._num_envs, self._max_actions, 3),
            dtype=torch.float32,
            device=self._device,
        )

        # 初次 reset 的姿态就用构造时的随机朝向；每个 episode 的换向，交给 reset_idx() 统一处理
        
        self.set_targets(self.all_indices)

        # 速度与动作的上一帧缓存（用于速度平滑/动作平滑）
        self.prev_speed = torch.zeros(self._num_envs, device=self._device)
        self.prev_actions = torch.zeros(self._num_envs, self._num_actions, device=self._device)

    def set_targets(self, env_ids: torch.Tensor):
        """
        Sets the targets for the task.

        Args:
            env_ids (torch.Tensor): the indices of the environments for which to set the targets.
        """

        num_sets = len(env_ids)
        env_long = env_ids.long()
        # Randomizes the position of the ball on the x y axis
        target_positions, target_orientation = self.task.get_goals(
            env_long, self.initial_pin_pos.clone(), self.initial_pin_rot.clone()
        )
        target_positions[env_long, 2] = torch.ones(num_sets, device=self._device) * 2.0
        # Apply the new goals
        if self._marker:
            self._marker.set_world_poses(
                target_positions[env_long],
                target_orientation[env_long],
                indices=env_long,
            )

        # 2) World -> Env：泊位中心在 env 系下的坐标（和 current_state["position"] 同一坐标系）
        bc_w = target_positions[env_long, :2]                  # [k,2] world
        env_origins = self._env_pos[env_long, :2]              # [k,2] world
        bc_env = bc_w - env_origins                            # [k,2] env

        # 3) 与可视化一致的朝向/尺寸（直接从已创建的 Berth 读，避免视觉与奖励不一致）
        yaw_list, L_list, W_list = [], [], []
        for e in env_long.tolist():
            yaw_list.append(float(self._berths[e].berth_orientation))  # radians
            L_list.append(float(self._berths[e].L_berth))
            W_list.append(float(self._berths[e].W_berth))
        berth_yaw = torch.tensor(yaw_list, device=self._device, dtype=torch.float32)  # [k]
        L = torch.tensor(L_list, device=self._device, dtype=torch.float32)            # [k]
        W = torch.tensor(W_list, device=self._device, dtype=torch.float32)            # [k]

        # 4) 预计算旋转：Env->Berth 用 R(-yaw)；同时给出"开口方向"在 Env 下的单位向量
        cy, sy = torch.cos(berth_yaw), torch.sin(berth_yaw)
        R_env2berth = torch.stack([cy,  sy,    # [[ cos,  sin],
                                   -sy, cy],   #  [ -sin, cos]]
                                  dim=-1).view(-1, 2, 2)                              # [k,2,2]
        open_dir_env = torch.stack([-sy, cy], dim=-1)                                  # [k,2] (+Y_B)

        # 5) 首次使用时在 task 上建好容器；然后只更新对应 env 的切片
        if not hasattr(self.task, "berth_center_xy_env"):
            self.task.berth_center_xy_env = torch.zeros(self._num_envs, 2,  device=self._device)
            self.task.berth_yaw          = torch.zeros(self._num_envs,     device=self._device)
            self.task.berth_size_LW      = torch.zeros(self._num_envs, 2,  device=self._device)
            self.task.R_env2berth        = torch.zeros(self._num_envs, 2, 2, device=self._device)
            self.task.open_dir_env       = torch.zeros(self._num_envs, 2,  device=self._device)

        self.task.berth_center_xy_env[env_long] = bc_env                    # [N,2] (env)
        self.task.berth_yaw[env_long]          = berth_yaw                 # [N]
        self.task.berth_size_LW[env_long]      = torch.stack([L, W], -1)   # [N,2] (L,W)
        self.task.R_env2berth[env_long]        = R_env2berth               # [N,2,2]
        self.task.open_dir_env[env_long]       = open_dir_env              # [N,2]

        # per-env wall thickness
        t_list = [float(self._berths[int(e)].wall_thickness) for e in env_long.tolist()]
        t_wall = torch.tensor(t_list, device=self._device, dtype=torch.float32)

        if not hasattr(self.task, "berth_wall_t"):
            self.task.berth_wall_t = torch.zeros(self._num_envs, device=self._device, dtype=torch.float32)

        self.task.berth_wall_t[env_long] = t_wall

        # 1) "代数一致性"自检
        if getattr(self, "debug_berth", False):
            # 1.1 无NaN/Inf & 形状
            t = "[DBG][set_targets]"
            k = env_long.numel()
            def _chk(name, x):
                bad = torch.isnan(x).any() | torch.isinf(x).any()
                if bad:
                    print(f"{t} {name} has NaN/Inf! shape={tuple(x.shape)}")
                return not bad
            ok = True
            ok &= _chk("berth_center_xy_env", self.task.berth_center_xy_env[env_long])
            ok &= _chk("berth_yaw",            self.task.berth_yaw[env_long])
            ok &= _chk("berth_size_LW",        self.task.berth_size_LW[env_long])
            ok &= _chk("R_env2berth",          self.task.R_env2berth[env_long])
            ok &= _chk("open_dir_env",         self.task.open_dir_env[env_long])

            # 1.2 正交矩阵与行列式≈+1
            R = self.task.R_env2berth[env_long]                 # [k,2,2]
            I = torch.eye(2, device=R.device).expand(k, 2, 2)
            ortho_err = (R @ R.transpose(1,2) - I).abs().max().item()
            detR = (R[:,0,0]*R[:,1,1] - R[:,0,1]*R[:,1,0])      # [k]
            det_err = (detR - 1.0).abs().max().item()

            # 1.3 "开口方向"一致性：R * open_dir_env ≈ [0,1]
            # （因为 R 是 Env->Berth，Env 中的开口单位向量应变到泊位坐标的 +Y 方向）
            o = self.task.open_dir_env[env_long]                # [k,2]
            o = o / (o.norm(dim=-1, keepdim=True) + 1e-8)
            ob = torch.bmm(R, o.unsqueeze(-1)).squeeze(-1)      # [k,2]
            open_err = (ob - torch.tensor([0.,1.], device=R.device)).abs().max().item()

            # 1.4 "中心点应在泊位原点"：把 (bc_env -> Berth) 应为 0
            bc_env = self.task.berth_center_xy_env[env_long]    # [k,2]
            zero_B = torch.bmm(R, torch.zeros(k,2,1, device=R.device)).squeeze(-1)
            # 注：对任意点 p_env，p_B = R*(p_env - bc_env)。带入 p_env=bc_env 得 0。
            # 这里就不重复计算了，只检查 R 是否数值稳定
            print(f"{t} k={k} | ortho_err={ortho_err:.2e} det_err={det_err:.2e} open_err={open_err:.2e}")

       

   

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        """
        Resets the environments with the given indices.

        Args:
            env_ids (torch.Tensor): the indices of the environments to be reset."""

        num_resets = len(env_ids)
        # Resets the counter of steps for which the goal was reached
        self.task.reset(env_ids)
        self.UF.generate_force(env_ids, num_resets)
        self.TD.generate_torque(env_ids, num_resets)
        self.MDD.randomize_masses(env_ids, num_resets)
        self.MDD.set_masses(self._heron.base, env_ids)
        # Resets hydrodynamic coefficients
        self.hydrodynamics.reset_coefficients(env_ids, num_resets)
        # Resets thruster randomization
        self.thrusters_dynamics.reset_thruster_randomization(env_ids, num_resets)
        
        # === 记录回合计数并重采样泊位朝向===
        self._berth_episode_count[env_ids] += 1
        
        # 2) 重采样泊位（含 set_yaw 更新可视化）
        self._resample_berths(env_ids)
        
        # Reset ILOS integral state
        self.ilos_zeta[env_ids] = 0.0
        
        # 3) 注入奖励/观测所需的泊位参数（_resample_berths 内部已调用，这里确保顺序）      
        # Randomizes the starting position of the platform within a disk around the target
        root_pos, root_rot = self.task.get_spawns(
            env_ids,
            self.initial_root_pos.clone(),
            self.initial_root_rot.clone(),
            self.step,
        )
        
        # 泊位门外扇区出生点覆写,保证船出生在泊位外且不撞墙
        root_pos, root_rot = self._override_spawn_to_door_sector(env_ids, root_pos, root_rot)
        
        root_pos[:, 2] = self.heron_zero_height + (
            -1 * self.heron_mass / (self.waterplane_area * self.water_density)
        )

        # Resets the states of the joints
        self.dof_pos[env_ids, :] = torch.zeros(
            (num_resets, self._heron.num_dof), device=self._device
        )
        self.dof_vel[env_ids, :] = 0
        # Sets the velocities to 0
        root_velocities = self.root_velocities.clone()

        root_velocities[env_ids] = 0
        # set root_velocities x, y direction randomly between -1.5m/s to +1.5m/s (in global)
        root_velocities[env_ids, 0] = (
            torch.rand(num_resets, device=self._device) * 3 - 1.5
        )
        root_velocities[env_ids, 1] = (
            torch.rand(num_resets, device=self._device) * 3 - 1.5
        )

        # apply resets
        self._heron.set_joint_positions(self.dof_pos[env_ids], indices=env_ids)
        self._heron.set_joint_velocities(self.dof_vel[env_ids], indices=env_ids)
        self._heron.set_world_poses(
            root_pos[env_ids], root_rot[env_ids], indices=env_ids
        )
        self._heron.set_velocities(root_velocities[env_ids], indices=env_ids)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        # --- reset berthing episode-local stats ---
        if hasattr(self, "steps_in_tolerance"):
            self.steps_in_tolerance[env_ids] = 0
        if hasattr(self, "berth_success_count"):
            self.berth_success_count[env_ids] = 0
        if hasattr(self, "berth_collision_count"):
            self.berth_collision_count[env_ids] = 0
        
        # --- 清零泊位奖励相关状态 ---
        eid = env_ids.long().to(self._device)
        self.prev_d_gate[eid] = 0.0
        self.prev_d_center[eid] = 0.0          
        self.prev_outside[eid] = True
        self.gate_crossed_flags[eid] = False
        self.dwell_steps[eid] = 0.0
        self.steps_in_tolerance[eid] = 0
        
        # 清零盆地逗留计数
        if hasattr(self, "basin_count"):
            self.basin_count[eid] = 0
        
        # 清零门口侧墙连续命中计数器
        if hasattr(self, "mouth_hit_count"):
            self.mouth_hit_count[eid] = 0
        if hasattr(self, "side_hit_count"):
            self.side_hit_count[eid] = 0
        
        # 速度与动作的上一帧缓存清零
        self.prev_speed[env_ids] = 0.0
        self.prev_actions[env_ids] = torch.zeros_like(self.prev_actions[env_ids])

        # 重置进门事件状态
        # 因为出生在门外，重置时要把历史擦掉，不然"刚出生就被认作已在门内"
        if hasattr(self, "prev_outside"):
            self.prev_outside[env_ids] = True  # 新生时默认在门外

        # fill `extras`
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"][key] = (
                torch.mean(self.episode_sums[key][env_ids]) / self._max_episode_length
            )
            self.episode_sums[key][env_ids] = 0.0

    def update_state_statistics(self) -> None:
        """
        Updates the statistics of the state of the training."""

        self.episode_sums["normed_linear_vel"] += torch.norm(
            self.current_state["linear_velocity"], dim=-1
        )
        self.episode_sums["normed_angular_vel"] += torch.abs(
            self.current_state["angular_velocity"]
        )
        self.episode_sums["actions_sum"] += torch.sum(self.actions, dim=-1)

    def calculate_metrics(self) -> None:
        """
        计算训练指标：基础奖励 + 靠泊奖励 + 惩罚项 + 课程化权重
        兼容 _penalties.compute_penalty 的两种签名（带/不带 step）。
        """
        import torch

        # --- 0) 训练进度 step ∈ [0,1] ---
        if not hasattr(self, "step"):
            self.step = 0.0

        # horizon_length 优先取 cfg；否则退回 _max_episode_length；再否则=1
        horizon = None
        if hasattr(self, "_task_cfg") and "env" in self._task_cfg:
            horizon = self._task_cfg["env"].get("horizon_length", None)
        if horizon is None:
            horizon = int(getattr(self, "_max_episode_length", 1))
        if horizon <= 0:
            horizon = 1

        # 累加归一化进度（可用于课程/惩罚退火）
        self.step += 1.0 / float(horizon)
        if self.step > 1.0:
            self.step = 1.0

        # --- 1) 基础奖励 ---
        base_reward = self.task.compute_reward(self.current_state, self.actions)

        # --- 2) 靠泊专项 shaping（我们在 Virtual 里实现的） ---
        use_improved = self._task_cfg.get("env", {}).get("berthing_reward", {}).get("use_improved", True)
        if use_improved:
            berth_reward = self.compute_berthing_reward_improved(self.current_state, self.actions)
        else:
            berth_reward = self.compute_berthing_reward(self.current_state, self.actions)

        # --- 3) 惩罚项（能耗/动作变化/角速变化等） ---
        penalties = 0.0
        if hasattr(self, "_penalties"):
            try:
                # 新签名：需要 step / 训练进度
                penalties = self._penalties.compute_penalty(self.current_state, self.actions, self.step)
            except TypeError:
                # 旧签名：仅 state, actions
                penalties = self._penalties.compute_penalty(self.current_state, self.actions)

        # 统一成张量并对齐设备/类型
        if torch.is_tensor(penalties):
            penalties = penalties.to(device=self.rew_buf.device, dtype=self.rew_buf.dtype)
        else:
            penalties = torch.full_like(base_reward, float(penalties))

        # --- 4) 读取/计算 奖励组合权重（支持静态 + 课程化） ---
        def _lerp(a: float, b: float, p: float) -> float:
            return float(a) + (float(b) - float(a)) * float(p)

        rw_cfg = {}
        if hasattr(self, "_task_cfg") and "env" in self._task_cfg:
            rw_cfg = self._task_cfg["env"].get("reward_weights", {}) or {}

        # 若提供 *_start/*_final 则启用课程化，否则用静态值，默认=1.0
        def _weight(name: str, default: float = 1.0) -> float:
            s_key, f_key = f"{name}_start", f"{name}_final"
            if s_key in rw_cfg or f_key in rw_cfg:
                a = float(rw_cfg.get(s_key, default))
                b = float(rw_cfg.get(f_key, default))
                return _lerp(a, b, float(self.step))
            return float(rw_cfg.get(name, default))

        w_base    = _weight("base",    1.0)
        w_berth   = _weight("berth",   1.0)
        w_penalty = _weight("penalty", 1.0)

        # --- 5) 合成总奖励 ---
        self.rew_buf[:] = (
            w_base   * base_reward +
            w_berth  * berth_reward +
            w_penalty* penalties
        )
        # 最终裁剪一层，避免价值网络炸
        self.rew_buf[:] = torch.clamp(self.rew_buf, -5.0, 5.0)

        # --- 6) 统计累计（episode sums） ---
        self.episode_sums = self.task.update_statistics(self.episode_sums)
        if hasattr(self, "_penalties"):
            self.episode_sums = self._penalties.update_statistics(self.episode_sums)
        self.update_state_statistics()

        # --- 7) 调试拆解 ---
        if getattr(self, "debug_rewards", False) and hasattr(self, "extras"):
            self.extras.setdefault("reward_breakdown", {})

            # episode 相对进度（每个 env 当前回合相对长度的平均）
            episode_progress = torch.tensor(0.0, device=self.rew_buf.device)
            if hasattr(self, "progress_buf"):
                max_len = int(getattr(self, "_max_episode_length", horizon))
                if max_len <= 0:
                    max_len = horizon
                episode_progress = (self.progress_buf.float() / float(max_len)).mean().to(self.rew_buf.device)

            self.extras["reward_breakdown"].update({
                # 组件均值（乘权重前）
                "base_reward_mean":   base_reward.mean(),
                "berth_reward_mean":  berth_reward.mean(),
                "penalties_mean":     penalties.mean(),
                # 组合后
                "total_reward_mean":  self.rew_buf.mean(),
                # 当前权重（便于排查课程化是否生效）
                "w_base":    torch.tensor(w_base,    device=self.rew_buf.device),
                "w_berth":   torch.tensor(w_berth,   device=self.rew_buf.device),
                "w_penalty": torch.tensor(w_penalty, device=self.rew_buf.device),
                # 进度监控
                "episode_progress_mean": episode_progress,
                "train_progress":        torch.tensor(self.step, device=self.rew_buf.device),
            })

    def is_done(self) -> None:
        """
        Checks if the episode is done."""

        # resets due to misbehavior
        ones = torch.ones_like(self.reset_buf)
        die = self.task.update_kills(self.step)

        # resets due to episode length
        # 先保留外部（奖励函数）已设置的 reset，再合并 is_done 的条件
        external = self.reset_buf.clone()
        self.reset_buf[:] = torch.where(
            self.progress_buf >= self._max_episode_length - 1, ones, die
        )
        self.reset_buf[:] = torch.maximum(self.reset_buf, external)

    def _debug_after_obs(self, state, mask=None):
        """
        几何一致性自检：检查船在泊位坐标中的位置是否合理
        """
        if not getattr(self, "debug_berth", False):
            return
        # 只挑几个env看
        idx = torch.arange(min(3, self._num_envs), device=self._device)
        if mask is not None:
            idx = idx[mask[idx] > 0]
            if idx.numel() == 0:
                idx = torch.arange(min(3, self._num_envs), device=self._device)

        p_env = state["position"][idx, :2]                  # [m] in Env
        bc_env = self.task.berth_center_xy_env[idx, :2]
        R = self.task.R_env2berth[idx]                      # Env->Berth
        d_env = p_env - bc_env
        p_B = torch.bmm(R, d_env.unsqueeze(-1)).squeeze(-1) # 船在泊位坐标

        L = self.task.berth_size_LW[idx, 0]
        W = self.task.berth_size_LW[idx, 1]
        halfL, halfW = 0.5*L, 0.5*W

        # 是否在"几何框"内（忽略墙厚/安全边）
        inside_x = (p_B[:,0].abs() <= halfW + 1e-4)
        inside_y = (p_B[:,1] >= -halfL - 1e-4) & (p_B[:,1] <= halfL + 1e-4)
        inside   = inside_x & inside_y

        # 到开口线 y=+L/2 的距离；正向为"朝开口方向前进"
        dist_gate = (halfL - p_B[:,1]).clamp(min=0.0)

        print("[DBG][after_obs]")
        for j, e in enumerate(idx.tolist()):
            print(f"  env {e}: p_B={p_B[j].tolist()} | L,W={float(L[j]):.2f},{float(W[j]):.2f} "
                  f"| inside={bool(inside[j])} | d_gate={float(dist_gate[j]):.2f}")

    def _debug_world_vs_env_distance(self, state):
        """
        运行时一致性自检：复核世界系距离和环境系距离是否一致
        """
        if not getattr(self, "debug_berth", False):
            return
        idx = torch.arange(min(3, self._num_envs), device=self._device)

        # 船世界坐标与env原点
        pos_w = self.root_pos[idx, :2]                     # [N,2] world
        env_w = self._env_pos[idx, :2]                     # [N,2] world
        pos_env = pos_w - env_w                            # [N,2] env（应等于state["position"]）

        bc_env = self.task.berth_center_xy_env[idx, :2]
       
        bc_w_recon = bc_env + env_w

        d_env = (pos_env - bc_env).norm(dim=-1)            # 距离(Env)
        d_w   = (pos_w  - bc_w_recon).norm(dim=-1)         # 距离(World via recon)

        print("[DBG][world_vs_env]")
        for j, e in enumerate(idx.tolist()):
            print(f"  env {e}: d_env={float(d_env[j]):.6f} | d_w={float(d_w[j]):.6f} | Δ={float((d_env[j]-d_w[j]).abs()):.3e}")

    def _sample_yaw(self, env_id: int) -> float:
        import math, torch
        mode = getattr(self, "berth_yaw_mode", "episode")
        if mode == "seeded":
            epi = int(self._berth_episode_count[env_id].item())
            g = torch.Generator(device='cpu')
            g.manual_seed(self.berth_seed * 10007 + env_id * 7919 + epi)
            return float(torch.rand((), generator=g).item() * 2*math.pi)
        # 默认：每次reset随即
        return float(torch.rand((), device=self._device).item() * 2*math.pi)

    def _resample_berths(self, env_ids: torch.Tensor):
        """按策略给这些 env 重采样泊位朝向，并同步 task.*（必须在 get_spawns 之前）。"""
        if getattr(self, "berth_eval_lock", False):
            return

        ids = env_ids.long().tolist()
        mode = getattr(self, "berth_yaw_mode", "episode")
        hold_k = int(getattr(self, "yaw_hold_k", 1))

        for e in ids:
            do_resample = True
            if mode == "once":
                do_resample = False
            elif mode == "success":
                # 只有上一回合成功才"允许换"
                bs = getattr(self, "berth_success_count",
                            torch.zeros(self._num_envs, dtype=torch.long, device=self._device))
                do_resample = bool(bs[e] > 0)

            # 二次门：hold_k（对所有允许换的模式都生效）
            if do_resample and hold_k > 1:
                epi = int(self._berth_episode_count[e].item())
                do_resample = (epi % hold_k == 0)

            if not do_resample:
                continue

            # 采样/固定
            if mode == "fixed" and getattr(self, "berth_fixed_yaw_deg", None) is not None:
                val = self.berth_fixed_yaw_deg
                if isinstance(val, (list, tuple)):
                    yaw_deg = float(val[e % len(val)])
                else:
                    yaw_deg = float(val)  # 支持标量
                yaw = math.radians(yaw_deg)
            else:
                yaw = self._sample_yaw(e)

            self._berths[e].set_yaw(yaw)  # 更新可视化三墙

            # 调试：读取后墙世界姿态的yaw，对比刚设置的berth_orientation
            if self.debug_berth:
                back_path = f"{self._berths[e].prim_path}/back_wall"
                xf = XFormPrim(prim_path=back_path, name=f"bw_{e}")
                p, q = xf.get_world_pose()            # q = [w,x, y, z]
                # world-yaw（Z朝上的欧拉角）
                num = 2.0 * (q[0]*q[3] + q[1]*q[2])
                den = 1.0 - 2.0 * (q[2]*q[2] + q[3]*q[3])
                yaw_w = math.degrees(math.atan2(num, den))
                print(f"[DBG][berth_vis] env={e} set_yaw(deg)={math.degrees(yaw):.1f}  back_wall_yaw_w(deg)={yaw_w:.1f}")

        # 同步奖励/重置所需的 task.*（中心/朝向/旋转矩阵/尺寸）
        self.set_targets(env_ids)
        
        # 调试打印：验证泊位朝向是否在变化
        if self.debug_berth:
            ids = env_ids.long().tolist()
            yaws_deg = [round(math.degrees(self._berths[e].berth_orientation), 1) for e in ids]
            epis = [int(self._berth_episode_count[e].item()) for e in ids]
            print(f"[DBG][resample] env={ids} epi={epis} yaw(deg)={yaws_deg}")

    def set_to_pose(
        self, env_ids: torch.Tensor, positions: torch.Tensor, heading: torch.Tensor
    ) -> None:
        """
        Sets the platform to a specific pose.
        TODO: Impose more iniiial conditions, such as linear and angular velocity.

        Args:
            env_ids (torch.Tensor): the indices of the environments for which to set the pose.
            positions (torch.Tensor): the positions of the platform.
            heading (torch.Tensor): the heading of the platform."""

        num_resets = len(env_ids)
        # Resets the counter of steps for which the goal was reached
        self.task.reset(env_ids)
        # Randomizes the starting position of the platform within a disk around the target
        root_pos = torch.zeros_like(self.root_pos)
        root_pos[env_ids, :2] = positions
        root_rot = torch.zeros_like(self.root_rot)
        root_rot[env_ids, :] = heading
        # Resets the states of the joints
        self.dof_pos[env_ids, :] = torch.zeros(
            (num_resets, self._heron.num_dof), device=self._device
        )
        self.dof_vel[env_ids, :] = 0
        # Sets the velocities to 0
        root_velocities = self.root_velocities.clone()
        root_velocities[env_ids] = 0

        # apply resets
        self._heron.set_joint_positions(self.dof_pos[env_ids], indices=env_ids)
        self._heron.set_joint_velocities(self.dof_vel[env_ids], indices=env_ids)
        self._heron.set_world_poses(
            root_pos[env_ids], root_rot[env_ids], indices=env_ids
        )
        self._heron.set_velocities(root_velocities[env_ids], indices=env_ids)

    def compute_berthing_reward_improved(self, state, actions):
        """
        改进版靠泊奖励（参考 HighwayEnv 风格：小而稳，连续平滑）
        * 使用 compute_adaptive_velocity_reward 作为速度条目（不要叠加其它速度逻辑）
        * 内部做了分量裁剪，最后在 calculate_metrics 里还会对总和再裁一层
        """
        import torch, math
        F = torch.nn.functional

        n, dev = self._num_envs, self._device

        # 0) 读取泊位几何（set_targets 已注入）
        if any(not hasattr(self.task, k) for k in ["berth_center_xy_env","R_env2berth","berth_yaw","berth_size_LW"]):
            return torch.zeros(n, device=dev)

        # 1) 基础几何：Env->Berth
        pos_env = state["position"][:, :2]
        vel_env = state.get("linear_velocity", torch.zeros((n, 2), device=dev))[:, :2]

        bc_env   = self.task.berth_center_xy_env           # [N,2]
        R_e2b    = self.task.R_env2berth                   # [N,2,2]
        LW       = self.task.berth_size_LW                 # [N,2] -> [L,W]
        berth_yaw= self.task.berth_yaw                     # [N]
        L, W = LW[:, 0], LW[:, 1]
        halfL, halfW = 0.5 * L, 0.5 * W

        rel_pos = pos_env - bc_env
        p_B = torch.einsum('nij,nj->ni', R_e2b, rel_pos)   # [N,2]
        v_B = torch.einsum('nij,nj->ni', R_e2b, vel_env)   # [N,2]
        xB, yB = p_B[:, 0], p_B[:, 1]
        vxB, vyB = v_B[:, 0], v_B[:, 1]

        #  先读取容差（供后面 in_basin_now、success 等统一使用）
        tp = self._task_cfg.get("env", {}).get("task_parameters", {})
        pos_tol_x = float(tp.get("position_tolerance_x", tp.get("position_tolerance", 0.3)))
        pos_tol_y = float(tp.get("position_tolerance_y", tp.get("position_tolerance", 0.3)))
        yaw_tol   = math.radians(float(tp.get("yaw_tolerance_deg", 10.0)))
        vel_tol   = float(tp.get("velocity_tolerance", 0.1))

        # === Boat envelope & wall geometry (all in berth frame) ===
        boat_half_w = 0.5 * float(self.box_width)
        boat_half_l = 0.5 * float(self.box_length)
        safety_buf  = 0.05

        tw    = getattr(self.task, "berth_wall_t", torch.full_like(halfW, 0.10))
        inner = halfW - tw
        inner_clear = inner - (boat_half_w + safety_buf)

        back_face  = -halfL + tw
        back_clear = back_face + (boat_half_l + safety_buf)

        # 2) 权重（可从 YAML 里覆盖）
        env_cfg = self._task_cfg.get("env", {})
        bw = env_cfg.get("berthing_reward_weights", {})
        w_dist      = float(bw.get("distance", 3.0))
        w_heading   = float(bw.get("heading", 2.0))
        w_velocity  = float(bw.get("velocity", 1.5))
        w_stability = float(bw.get("stability", 1.0))
        w_collision = float(bw.get("collision", 10.0))
        w_danger    = float(bw.get("danger", 0.6))      # 危险度软约束权重（小权重）

        # 3) 门判定与距离
        outside = (yB > halfL)
        d_gate = (yB - halfL).clamp(min=0.0)
        r_dist_out = -1.0 + torch.exp(-d_gate / 5.0)       # [-1,0]

        d_center = torch.sqrt(xB * xB + yB * yB + 1e-6)
        sigma2 = (halfL**2 + halfW**2) / 4.0
        center_score = torch.exp(-d_center**2 / (2.0 * sigma2))    # [0,1]

        gate_smooth = torch.sigmoid((halfL - yB) / 2.0)            # 2m 过渡区
        r_dist = (1 - gate_smooth) * r_dist_out + gate_smooth * center_score  # ≈[-1,1]

        # 4) 朝向（门内主用；门外漏斗继续沿用同样的 d_yaw）
        boat_yaw = self._extract_boat_yaw(state, v_B)
        psi_path = berth_yaw - math.pi * 0.5  # 泊位中心线方向（-Y_B）

        # --- ILOS 与控制侧保持一致 ---
        # ILOS 积分态：训练初期可能尚未初始化，这里容错一下
        zeta = getattr(self, "ilos_zeta", torch.zeros_like(xB))
        arg  = - self.kp_ilos * xB - self.ki_ilos * zeta
        chi_d = psi_path + torch.atan(arg)  # kp=ki=0 时退化为 psi_path

        # --- β：用泊位系船速估计蟹行角（更贴近真实侧漂） ---
        # 侧漂角：沿 -Y_B（驶入库内方向）观察到的横飘角；vyB<0 为向内
        beta = torch.atan2(vxB, (-vyB).clamp(min=1e-3))

        # k_beta 可走配置：优先取 self.as_k_beta；没有就回退到 action-safety 的 current_heading_gain
        k_beta = float(getattr(self, "as_k_beta",
                       getattr(self, "as_current_heading_gain", 0.8)))

        # --- 渐变：门口保留 β（便于斜向入门），越靠后墙越回到纯 psi_path ---
        depth = ((halfL - yB) / (halfL + 1e-6)).clamp(0.0, 1.0)  # 0:门口, 1:靠后墙
        blend = depth * depth

        # 目标朝向：在门口用 chi_d + k_beta*beta，靠后墙用 psi_path
        desired_yaw_in = torch.atan2(
            torch.sin(chi_d + k_beta * beta) * (1.0 - blend) + torch.sin(psi_path) * blend,
            torch.cos(chi_d + k_beta * beta) * (1.0 - blend) + torch.cos(psi_path) * blend
        )

        # 统一 d_yaw：门内用 desired_yaw_in；门外漏斗也用相同 d_yaw 计算
        d_yaw = (boat_yaw - desired_yaw_in + math.pi) % (2 * math.pi) - math.pi
        r_heading = torch.where(~outside, 0.5 * (1.0 + torch.cos(d_yaw)), torch.zeros_like(d_yaw))

        # --- 4b) 门外对齐漏斗：门外 [halfL, halfL+1.2] 逐步加强对齐与居中 ---
        funnel_zone = (yB > halfL) & (yB < (halfL + 1.2))
        w_funnel    = ((halfL + 1.2) - yB).clamp(min=0.0) / 1.2

        # 朝向对齐（与门内一致的余弦型，直接复用 d_yaw）
        r_yaw_out   = w_funnel * 0.5 * (1.0 + torch.cos(d_yaw))

        # 横向居中：用"净半宽 = halfW - 墙厚"归一，避免墙厚带来的偏差
        wall_t = getattr(self.task, "berth_wall_t", torch.full_like(halfW, 0.10))
        innerW = (halfW - wall_t).clamp(min=0.05)
        r_lat_out = - w_funnel * (torch.abs(xB) / (innerW + 1e-6)).clamp(max=1.5)

        # 5) 速度条目（采用自适应速度版本；不要叠加其它速度奖励）
        r_velocity, target_speed = self.compute_adaptive_velocity_reward(
            state, xB, yB, vxB, vyB, d_center, outside, halfL, halfW, desired_yaw_in
        )
        
        # 修正"径向速度"在中心的数值稳定性
        dx, dy = -xB, -yB  # 指向中心的位移
        eps = 1e-3
        den = torch.clamp(torch.sqrt(dx*dx + dy*dy), min=eps)
        u_vec_x = dx / den  # 单位径向向量的x分量
        u_vec_y = dy / den  # 单位径向向量的y分量
        v_rad = vxB * u_vec_x + vyB * u_vec_y  # 径向速度（数值稳定）

        # 6) 稳定性（动作平滑+角速度稳定；门内更看重）
        if hasattr(self, 'prev_actions'):
            action_diff = torch.norm(actions - self.prev_actions, dim=-1)
            r_smooth = torch.exp(-2.0 * action_diff)               # [0,1]
        else:
            r_smooth = torch.ones(n, device=dev)
        self.prev_actions = actions.clone()

        ang = state.get("angular_velocity", torch.zeros(n, device=dev))
        # update_state 提供的是标量 yaw rate（N,），直接用
        yaw_rate = ang if ang.ndim == 1 else ang[:, -1]
        r_angular = torch.exp(-2.0 * torch.abs(yaw_rate))          # [0,1]
        r_stability = torch.where(~outside, 0.7 * r_smooth + 0.3 * r_angular, r_smooth)
        # 并入门外对齐漏斗（小权重叠加）
        r_stability = r_stability + 0.6 * r_yaw_out + 0.4 * r_lat_out

        # 7) 软边碰撞：走廊内对侧墙/后墙软惩罚（按外廓）
        soft_margin = 0.15
        # ---- collision-like soft penalties (use boat envelope) ----
        # 侧墙：当“中心|xB|+船半宽”越过 “内壁-软边” 才开始罚
        side_pen = (torch.abs(xB) + boat_half_w - (inner - soft_margin)).clamp(min=0.0)
        mask_side = (~outside) & (side_pen > 0)

        # 后墙：当“中心yB”小于 “内壁 + 船半长 + 软边” 才开始罚
        back_pen  = ((back_face + boat_half_l + soft_margin) - yB).clamp(min=0.0)
        mask_back = (~outside) & (yB <= (back_face + boat_half_l + soft_margin))

        # 继续用指数放大
        back_pen = torch.exp(back_pen / soft_margin) - 1.0

        r_collision = torch.zeros_like(xB)
        r_collision[mask_side] -= torch.clamp(side_pen[mask_side], max=1.0)
        r_collision[mask_back] -= torch.clamp(back_pen[mask_back], max=1.0)

        # ==== Side-wall proximity & lateral-velocity penalties ====
        side_margin = 0.35   # 侧墙缓冲带（m），和走廊/后墙 margin 类似
        # 只在走廊内生效（避免在开口外/库外干扰）
        in_corridor = ( (yB > -halfL) & (yB < (halfL - self.as_entrance_guard)) )

        # 接近侧墙的归一化接近度：离墙 < side_margin 时从0→1
        # 软墙几何基准：以内壁为基线（减去墙厚），并考虑船半宽
        prox_side = ((torch.abs(xB) + boat_half_w) - (inner - side_margin)) / (side_margin + 1e-6)
        prox_side = torch.clamp(prox_side, 0.0, 1.0)

        # (a) 几何接近惩罚（平方更平滑）
        r_side_geo = -3.0 * (prox_side ** 2) * in_corridor.float()

        # (b) 侧向速度惩罚（越贴墙，越不许有大 |vxB|）
        r_side_v = -1.5 * torch.abs(vxB) * prox_side * in_corridor.float()
        # 门口外侧 0~entrance_guard 区间也给一点点软惩罚，避免沿墙摩擦前行
        entrance_guard = float(getattr(self, "as_entrance_guard", 0.05))
        mouth_outer = (outside & (yB < (halfL + entrance_guard)))
        r_side_geo = r_side_geo + (-1.5 * (prox_side ** 2) * 0.4) * mouth_outer.float()
        r_side_v   = r_side_v   + (-1.5 * torch.abs(vxB) * prox_side * 0.3) * mouth_outer.float()

        # === Side-wall proximity & lateral-velocity penalties ===
        # 退火（0→1），先用 step；后续如果要更慢，把 1.0 改成 0.5
        anneal = torch.clamp(torch.tensor(self.step, device=self._device), 0.0, 1.0)
        r_side_geo = r_side_geo * anneal
        r_side_v   = r_side_v   * anneal

        # === New: Danger (soft proximity) — 只作为软约束，不并入 collision ===
        bw = self._task_cfg["env"].get("berthing_reward_weights", {})
        danger_margin = float(bw.get("danger_margin", 0.20))  # 可在 YAML 覆盖
        # 距离侧墙的余量（米）
        # 软墙几何基准：以内壁为基线（减去墙厚）
        tw = getattr(self.task, "berth_wall_t", torch.full_like(halfW, 0.10))
        inner = halfW - tw
        d_side = (inner - torch.abs(xB)).clamp(min=0.0)
        # 危险度：余量<margin 时从 0→1
        danger = torch.clamp((danger_margin - d_side) / (danger_margin + 1e-6), 0.0, 1.0)
        # 软约束：几何平方 + 侧向速度耦合（小权重、门内生效）
        r_danger = - (0.5 * danger * danger + 0.3 * torch.abs(vxB) * danger) * in_corridor.float()
        # 给个温和上限，绝不爆负
        r_danger = torch.clamp(r_danger, min=-0.8)

        # 8) 进度差（门外更重）
        progress_reward = torch.zeros(n, device=dev)
        if hasattr(self, 'prev_d_gate') and hasattr(self, 'prev_d_center'):
            gate_progress = (self.prev_d_gate - d_gate).clamp(min=-0.5, max=0.5)
            center_progress = (self.prev_d_center - d_center).clamp(min=-0.5, max=0.5)
            progress_reward = torch.where(outside, 2.0 * gate_progress, center_progress)

        # 9) 事件（小幅度 + 渐进成功）
        r_events = torch.zeros(n, device=dev)
        
        # --- 后墙滞后触发保护带：进入中心盆地后累计 ≥5 步，且确实向后越线再触发 ---
        # 读取/维护盆地逗留步数
        in_basin_now = (~outside) & (torch.abs(xB) <= (pos_tol_x + 0.25)) & (torch.abs(yB) <= (pos_tol_y + 0.25))
        basin_count  = getattr(self, "basin_count", torch.zeros_like(xB, dtype=torch.int32))
        basin_count  = torch.where(in_basin_now, basin_count + 1, torch.zeros_like(basin_count))
        self.basin_count = basin_count

        # 仅当"已经在盆地至少5步"后，且确实以船尾越线再触发失败
        back_guard = (basin_count >= 5) & (~outside) \
                     & ((yB - boat_half_l) < (-halfL + tw + 0.05)) \
                     & (vyB < -0.10)
        r_events[back_guard] += -12.0
        if hasattr(self, "reset_buf"):
            self.reset_buf[back_guard] = 1
        
        # —— 在函数一开始就做个快照（放在第一次用到 prev_outside 之前）——
        prev_out = getattr(self, 'prev_outside', torch.ones_like(outside))
        
        # 用旧的 prev_out 做穿门检测
        gate_crossed = prev_out & (~outside)
        if hasattr(self, 'gate_crossed_flags'):
            new_cross = gate_crossed & (~self.gate_crossed_flags)
            r_events[new_cross] += 2.0
            self.gate_crossed_flags |= gate_crossed

        pos_ok = (~outside) & (torch.abs(xB) <= pos_tol_x) & (torch.abs(yB) <= pos_tol_y)
        yaw_ok = (torch.abs(d_yaw) <= yaw_tol)
        vel_ok = (torch.sqrt(vxB*vxB + vyB*vyB) <= vel_tol)
        success = pos_ok & yaw_ok & vel_ok

        if not hasattr(self, 'steps_in_tolerance'):
            self.steps_in_tolerance = torch.zeros(n, dtype=torch.long, device=dev)
        self.steps_in_tolerance = torch.where(success,
                                              self.steps_in_tolerance + 1,
                                              torch.zeros_like(self.steps_in_tolerance))
        # --- config 驱动的保持奖励（防止奖励尺度突变） ---
        tp = self._task_cfg.get("env", {}).get("task_parameters", {})
        rp = self._task_cfg.get("env", {}).get("reward_parameters", {})
        H = int(tp.get("kill_after_n_steps_in_tolerance", 50))      # 默认 50
        hold_gain = float(rp.get("success_hold_gain", 5.0))          # 默认 5.0（原先相当于 10.0）

        r_events += torch.clamp(self.steps_in_tolerance.float() / max(H, 1), max=1.0) * hold_gain
        
        # --- 黏性成功：一旦进过达标区，不允许往后墙方向逃离 ---
        pos_tol_x = float(tp.get("position_tolerance_x", 0.5))
        pos_tol_y = float(tp.get("position_tolerance_y", 0.5))

        # 计算在上方已有：
        pos_ok = (~outside) & (torch.abs(xB) <= pos_tol_x) & (torch.abs(yB) <= pos_tol_y)
        
        was_ok = getattr(self, "prev_ok", torch.zeros_like(pos_ok, dtype=torch.bool))
        leaving_back = was_ok & (yB < -(pos_tol_y + 0.15))   # 从中心区向后越线

        r_events[leaving_back] += -20.0
        if hasattr(self, "reset_buf"):
            self.reset_buf[leaving_back] = 1

        self.prev_ok = pos_ok.clone()  # ← 由 success 改为 pos_ok

        # 在"9) 事件"逻辑里 success 判定之后追加：
        tp = self._task_cfg.get("env", {}).get("task_parameters", {})
        hold_N   = int(tp.get("kill_after_n_steps_in_tolerance", 50))
        kill_dist = float(tp.get("kill_dist", 70.0))
        # 这里直接用上面已累计好的 steps_in_tolerance
        just_success = self.steps_in_tolerance >= hold_N

        # === Side/back fail lines unified by boat envelope ===
        eps = 0.02
        # 1) 侧墙内侧命中（库内）——按船体包络，越内壁 2cm 以上算一次"命中"
        hit_side_inside = ((~outside) & (yB >= -halfL)
                           & (torch.abs(xB) >= (inner_clear + eps)))

        # 只在"正在向墙推进"时计入（更有物理意义）
        push_into_wall  = (xB * vxB) > 0.05  # 阈值略大于0，避免数值抖动
        hit_side_inside = hit_side_inside & push_into_wall

        # 连续 K 帧才失败（避免单帧误杀）
        S_side = 6  # 可配
        side_cnt = getattr(self, "side_hit_count",
                           torch.zeros_like(xB, dtype=torch.int32))
        side_cnt = torch.where(hit_side_inside,
                               side_cnt + 1, torch.zeros_like(side_cnt))
        self.side_hit_count = side_cnt

        # 2) 门口侧墙失败（库外、离门内侧 guard 范围内）——"几何越界即记一次命中"
        guard = float(getattr(self, "as_entrance_guard", 0.05))
        in_mouth_band = (outside & (yB <= (halfL + guard)))
        penetrate     = (torch.abs(xB) >= (inner_clear + eps))
        hit_mouth     = in_mouth_band & penetrate

        # 连续命中计数：可配 mouth_fail_hits（默认 3 帧，更稳）
        rp = self._task_cfg.get("env", {}).get("reward_parameters", {})
        H_mouth = int(rp.get("mouth_fail_hits", 3))

        cnt = getattr(self, "mouth_hit_count", torch.zeros_like(xB, dtype=torch.int32))
        cnt = torch.where(hit_mouth, cnt + 1, torch.zeros_like(cnt))
        self.mouth_hit_count = cnt

        fail_side_mouth_persist = (cnt >= H_mouth)

        # 单帧命中只在"新发生"的那一帧记一次小罚，避免连续爆负
        prev_hit_mouth = getattr(self, "prev_hit_mouth",
                                 torch.zeros_like(hit_mouth, dtype=torch.bool))
        new_hit_mouth  = hit_mouth & (~prev_hit_mouth)
        r_events = torch.where(new_hit_mouth, r_events - 3.0, r_events)
        self.prev_hit_mouth = hit_mouth.clone()

        # 库内侧墙"持续命中"在 fail_lateral 触发时已有终止，不再每帧额外-10

        # 失败条件
        fail_lateral = (self.side_hit_count >= S_side)
        # 3) 后墙失败（中心 y 低于允许最小值 back_clear）
        fail_back    = (yB < back_clear)
        fail_far     = (d_center > kill_dist)
        fail_mask    = fail_lateral | fail_side_mouth_persist | fail_back | fail_far

        # 给一点小的终止事件值（不大），同时设置 reset_buf
        r_events[just_success] += 10.0   # 与你的渐进成功最高 10 一致
        r_events[fail_mask]    += -5.0   # 轻量失败

        # 成功计数+1（用于success模式的泊位重采样）
        if hasattr(self, "berth_success_count"):
            self.berth_success_count[just_success] += 1

        if hasattr(self, "reset_buf"):
            self.reset_buf[(just_success | fail_mask).view(-1)] = 1

        # 10) 合成与裁剪（本条目内部裁一层）
        r_total = (
            w_dist * r_dist +
            w_heading * r_heading +
            w_velocity * r_velocity +
            w_stability * r_stability +
            w_collision * r_collision +   # 现在只承载"软/硬边界穿越"的barrier类项
            w_danger * r_danger +         # << 新增：危险度软约束（小权重）
            progress_reward +
            r_events
        )
        r_total = torch.clamp(r_total, -5.0, 5.0)

        # 11) 合成前/后，最后再统一更新这些缓存
        self.prev_d_gate   = d_gate.clone()
        self.prev_d_center = d_center.clone()
        self.prev_outside  = outside.clone()

        if getattr(self, "debug_rewards", False):
            self.extras.setdefault("berthing_metrics", {}).update({
                "r_dist": r_dist.mean(),
                "r_heading": r_heading.mean(),
                "r_velocity": r_velocity.mean(),
                "r_stability": r_stability.mean(),
                "r_collision": r_collision.mean(),
                "danger_mean": danger.mean(),
                "r_danger": r_danger.mean(),
                "progress": progress_reward.mean(),
                "target_speed_mean": target_speed.mean(),
                "r_total": r_total.mean(),
            })

        return r_total
