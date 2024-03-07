__author__ = "Antoine Richard, Junghwan Ro, Matteo El Hariry"
__copyright__ = (
    "Copyright 2023, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Junghwan Ro"
__email__ = "jro37@gatech.edu"
__status__ = "development"

import torch
from dataclasses import dataclass

EPS = 1e-6  # small constant to avoid divisions by 0 and log(0)


@dataclass
class GoToXYReward:
    """ "
    Reward function and parameters for the GoToXY task."""

    reward_mode: str = "exponential"
    position_scale: float = 1.0
    heading_scale: float = 1.0
    exponential_reward_coeff: float = 0.25

    def __post_init__(self) -> None:
        """
        Checks that the reward parameters are valid."""

        assert self.reward_mode.lower() in [
            "linear",
            "square",
            "exponential",
        ], "Linear, Square and Exponential are the only currently supported mode."

    def compute_reward(
        self,
        current_state: torch.Tensor,
        actions: torch.Tensor,
        position_error: torch.Tensor,
        prev_position_error: torch.Tensor,
        heading_error: torch.Tensor,
        prev_heading_error: torch.Tensor,
    ) -> torch.Tensor:
        """
        Defines the function used to compute the reward for the GoToXY task."""

        if self.reward_mode.lower() == "linear":
            position_reward = self.position_scale * (
                prev_position_error - position_error
            ) + self.heading_scale * (prev_heading_error - heading_error)
        elif self.reward_mode.lower() == "square":
            position_reward = self.position_scale * (
                prev_position_error.pow(2) - position_error.pow(2)
            ) + self.heading_scale * (prev_heading_error.pow(2) - heading_error.pow(2))
        elif self.reward_mode.lower() == "exponential":
            position_reward = self.position_scale * (
                torch.exp(-position_error / self.exponential_reward_coeff)
                - torch.exp(-prev_position_error / self.exponential_reward_coeff)
            ) + self.heading_scale * (
                torch.exp(-heading_error / self.exponential_reward_coeff)
                - torch.exp(-prev_heading_error / self.exponential_reward_coeff)
            )
        else:
            raise ValueError("Unknown reward type.")
        return position_reward


@dataclass
class GoToPoseReward:
    """
    Reward function and parameters for the GoToPose task."""

    position_reward_mode: str = "exponential"
    heading_reward_mode: str = "exponential"
    position_exponential_reward_coeff: float = 0.25
    heading_exponential_reward_coeff: float = 0.25
    position_scale: float = 1.0
    heading_scale: float = 5.0
    sig_gain: float = 3.0

    def sigmoid(self, x, gain=3.0, offset=2):
        """Sigmoid function for dynamic weighting."""
        return 1 / (1 + torch.exp(-gain * (x - offset)))

    def __post_init__(self) -> None:
        """
        Checks that the reward parameters are valid."""

        assert self.position_reward_mode.lower() in [
            "linear",
            "square",
            "exponential",
        ], "Linear, Square and Exponential are the only currently supported mode."
        assert self.heading_reward_mode.lower() in [
            "linear",
            "square",
            "exponential",
        ], "Linear, Square and Exponential are the only currently supported mode."

    def compute_reward(
        self,
        current_state,
        actions: torch.Tensor,
        position_error: torch.Tensor,
        heading_error: torch.Tensor,
    ) -> None:
        """
        Defines the function used to compute the reward for the GoToPose task.
        d + k^d * h
        k^d is weighting term, where k is 0<k<1
        """
        # Adjust heading reward based on distance to goal
        heading_weight_factor = 1.0 - self.sigmoid(position_error, self.sig_gain)

        if self.position_reward_mode.lower() == "linear":
            position_reward = self.position_scale * (1.0 / (1.0 + position_error))
        elif self.position_reward_mode.lower() == "square":
            position_reward = self.position_scale * (
                1.0 / (1.0 + position_error * position_error)
            )
        elif self.position_reward_mode.lower() == "exponential":
            position_reward = self.position_scale * torch.exp(
                -position_error / self.position_exponential_reward_coeff
            )
        else:
            raise ValueError("Unknown reward type.")

        if self.heading_reward_mode.lower() == "linear":
            heading_reward = (
                heading_weight_factor
                * self.heading_scale
                * (1.0 / (1.0 + heading_error))
            )
        elif self.heading_reward_mode.lower() == "square":
            heading_reward = (
                heading_weight_factor
                * self.heading_scale
                * (1.0 / (1.0 + heading_error * heading_error))
            )
        elif self.heading_reward_mode.lower() == "exponential":
            heading_reward = (
                heading_weight_factor
                * self.heading_scale
                * torch.exp(-heading_error / self.heading_exponential_reward_coeff)
            )
        else:
            raise ValueError("Unknown reward type.")
        return position_reward, heading_reward


@dataclass
class KeepXYReward:
    """ "
    Reward function and parameters for the KeepXY task."""

    reward_mode: str = "linear"
    exponential_reward_coeff: float = 0.25

    def __post_init__(self) -> None:
        """
        Checks that the reward parameters are valid."""

        assert self.reward_mode.lower() in [
            "linear",
            "square",
            "exponential",
        ], "Linear, Square and Exponential are the only currently supported mode."

    def compute_reward(
        self,
        current_state: torch.Tensor,
        actions: torch.Tensor,
        position_error: torch.Tensor,
    ) -> torch.Tensor:
        """
        Defines the function used to compute the reward for the KeepXY task."""

        if self.reward_mode.lower() == "linear":
            position_reward = 1.0 / (1.0 + position_error)
        elif self.reward_mode.lower() == "square":
            position_reward = 1.0 / (1.0 + position_error * position_error)
        elif self.reward_mode.lower() == "exponential":
            position_reward = torch.exp(-position_error / self.exponential_reward_coeff)
        else:
            raise ValueError("Unknown reward type.")
        return position_reward


@dataclass
class TrackXYVelocityReward:
    """
    Reward function and parameters for the TrackXYVelocity task."""

    reward_mode: str = "exponential"
    exponential_reward_coeff: float = 0.25

    def __post_init__(self) -> None:
        """
        Checks that the reward parameters are valid."""

        assert self.reward_mode.lower() in [
            "linear",
            "square",
            "exponential",
        ], "Linear, Square and Exponential are the only currently supported mode."

    def compute_reward(
        self,
        current_state: torch.Tensor,
        actions: torch.Tensor,
        velocity_error: torch.Tensor,
    ) -> None:
        """
        Defines the function used to compute the reward for the TrackXYVelocity task."""

        if self.reward_mode.lower() == "linear":
            velocity_reward = 1.0 / (1.0 + velocity_error)
        elif self.reward_mode.lower() == "square":
            velocity_reward = 1.0 / (1.0 + velocity_error * velocity_error)
        elif self.reward_mode.lower() == "exponential":
            velocity_reward = torch.exp(-velocity_error / self.exponential_reward_coeff)
        else:
            raise ValueError("Unknown reward type.")
        return velocity_reward


@dataclass
class TrackXYOVelocityReward:
    """
    Reward function and parameters for the TrackXYOVelocity task."""

    linear_reward_mode: str = "exponential"
    angular_reward_mode: str = "exponential"
    linear_exponential_reward_coeff: float = 0.25
    angular_exponential_reward_coeff: float = 0.25
    linear_scale: float = 1.0
    angular_scale: float = 1.0

    def __post_init__(self) -> None:
        """
        Checks that the reward parameters are valid."""

        assert self.linear_reward_mode.lower() in [
            "linear",
            "square",
            "exponential",
        ], "Linear, Square and Exponential are the only currently supported mode."
        assert self.angular_reward_mode.lower() in [
            "linear",
            "square",
            "exponential",
        ], "Linear, Square and Exponential are the only currently supported mode."

    def compute_reward(
        self,
        current_state,
        actions: torch.Tensor,
        linear_velocity_error: torch.Tensor,
        angular_velocity_error: torch.Tensor,
    ) -> None:
        """
        Defines the function used to compute the reward for the TrackXYOVelocity task.
        """

        if self.linear_reward_mode.lower() == "linear":
            linear_reward = 1.0 / (1.0 + linear_velocity_error) * self.linear_scale
        elif self.linear_reward_mode.lower() == "square":
            linear_reward = 1.0 / (1.0 + linear_velocity_error**2) * self.linear_scale
        elif self.linear_reward_mode.lower() == "exponential":
            linear_reward = (
                torch.exp(-linear_velocity_error / self.linear_exponential_reward_coeff)
                * self.linear_scale
            )
        else:
            raise ValueError("Unknown reward type.")

        if self.angular_reward_mode.lower() == "linear":
            angular_reward = 1.0 / (1.0 + angular_velocity_error) * self.angular_scale
        elif self.angular_reward_mode.lower() == "square":
            angular_reward = (
                1.0 / (1.0 + angular_velocity_error**2) * self.angular_scale
            )
        elif self.angular_reward_mode.lower() == "exponential":
            angular_reward = (
                torch.exp(
                    -angular_velocity_error / self.angular_exponential_reward_coeff
                )
                * self.angular_scale
            )
        else:
            raise ValueError("Unknown reward type.")
        return linear_reward, angular_reward


@dataclass
class Penalties:
    """
    Metaclass to compute penalties for the tasks."""

    penalize_linear_velocities: bool = False
    penalize_linear_velocities_fn: str = (
        "lambda x,step : -torch.norm(x, dim=-1)*c1 + c2"
    )
    penalize_linear_velocities_c1: float = 0.01
    penalize_linear_velocities_c2: float = 0.0
    penalize_angular_velocities: bool = False
    penalize_angular_velocities_fn: str = "lambda x,step : -torch.abs(x)*c1 + c2"
    penalize_angular_velocities_c1: float = 0.01
    penalize_angular_velocities_c2: float = 0.0
    penalize_energy: bool = False
    penalize_energy_fn: str = "lambda x,step : -torch.abs(x)*c1 + c2"
    penalize_energy_c1: float = 0.01
    penalize_energy_c2: float = 0.0

    def __post_init__(self):
        """
        Converts the string functions into python callable functions."""
        self.penalize_linear_velocities_fn = eval(self.penalize_linear_velocities_fn)
        self.penalize_angular_velocities_fn = eval(self.penalize_angular_velocities_fn)
        self.penalize_energy_fn = eval(self.penalize_energy_fn)

    def compute_penalty(
        self, state: torch.Tensor, actions: torch.Tensor, step: int
    ) -> torch.Tensor:
        """
        Computes the penalties for the task."""

        # Linear velocity penalty
        if self.penalize_linear_velocities:
            self.linear_vel_penalty = self.penalize_linear_velocities_fn(
                state["linear_velocity"],
                torch.tensor(step, dtype=torch.float32, device=actions.device),
            )
        else:
            self.linear_vel_penalty = torch.zeros(
                [actions.shape[0]], dtype=torch.float32, device=actions.device
            )
        # Angular velocity penalty
        if self.penalize_angular_velocities:
            self.angular_vel_penalty = self.penalize_angular_velocities_fn(
                state["angular_velocity"],
                torch.tensor(step, dtype=torch.float32, device=actions.device),
            )
        else:
            self.angular_vel_penalty = torch.zeros(
                [actions.shape[0]], dtype=torch.float32, device=actions.device
            )
        # Energy penalty
        if self.penalize_energy:
            self.energy_penalty = self.penalize_energy_fn(
                torch.sum(actions, -1),
                torch.tensor(step, dtype=torch.float32, device=actions.device),
            )
        else:
            self.energy_penalty = torch.zeros(
                [actions.shape[0]], dtype=torch.float32, device=actions.device
            )

        # print penalties
        # print("linear_vel_penalty: ", self.linear_vel_penalty)
        # print("angular_vel_penalty: ", self.angular_vel_penalty)
        # print("energy_penalty: ", self.energy_penalty)

        return self.linear_vel_penalty + self.angular_vel_penalty + self.energy_penalty

    def get_stats_name(self) -> list:
        """
        Returns the names of the statistics to be computed."""

        names = []
        if self.penalize_linear_velocities:
            names.append("linear_vel_penalty")
        if self.penalize_angular_velocities:
            names.append("angular_vel_penalty")
        if self.penalize_energy:
            names.append("energy_penalty")
        return names

    def update_statistics(self, stats: dict) -> dict:
        """
        Updates the training statistics."""

        if self.penalize_linear_velocities:
            stats["linear_vel_penalty"] += self.linear_vel_penalty
        if self.penalize_angular_velocities:
            stats["angular_vel_penalty"] += self.angular_vel_penalty
        if self.penalize_energy:
            stats["energy_penalty"] += self.energy_penalty
        return stats
