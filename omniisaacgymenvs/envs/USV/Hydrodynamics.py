import torch
import pytorch3d.transforms
from omniisaacgymenvs.envs.USV.Utils import *


class HydrodynamicsObject:
    def __init__(
        self,
        num_envs,
        device,
        water_density,
        gravity,
        drag_coefficients,
        linear_damping,
        quadratic_damping,
        linear_damping_forward_speed,
        offset_linear_damping,
        offset_lin_forward_damping_speed,
        offset_nonlin_damping,
        scaling_damping,
        offset_added_mass,
        scaling_added_mass,
        alpha,
        last_time,
    ):
        self._num_envs = num_envs
        self.device = device
        self.drag = torch.zeros(
            (self._num_envs, 6), dtype=torch.float32, device=self.device
        )

        # damping parameters
        self.drag_coefficients = torch.tensor(
            [drag_coefficients], device=self.device
        )  # 1*6
        self.linear_damping = torch.tensor([linear_damping], device=self.device)  # 1*6
        self.quadratic_damping = torch.tensor(
            [quadratic_damping], device=self.device
        )  # 1*6
        self.linear_damping_forward_speed = torch.tensor(
            linear_damping_forward_speed, device=self.device
        )
        self.offset_linear_damping = offset_linear_damping
        self.offset_lin_forward_damping_speed = offset_lin_forward_damping_speed
        self.offset_nonlin_damping = offset_nonlin_damping
        self.scaling_damping = scaling_damping

        # coriolis
        self._Ca = torch.zeros([6, 6], device=self.device)
        self.added_mass = torch.zeros([num_envs, 6], device=self.device)
        self.offset_added_mass = offset_added_mass
        self.scaling_added_mass = scaling_added_mass

        # acceleration
        self.alpha = alpha
        self._filtered_acc = torch.zeros([6], device=self.device)
        self._last_time = last_time
        self._last_vel_rel = torch.zeros([6], device=self.device)

        return

    def ComputeDampingMatrix(self, vel):
        """
        // From Antonelli 2014: the viscosity of the fluid causes
        // the presence of dissipative drag and lift forces on the
        // body. A common simplification is to consider only linear
        // and quadratic damping terms and group these terms in a
        // matrix Drb
        """

        lin_damp = (
            self.linear_damping
            + self.offset_linear_damping
            - (
                self.linear_damping_forward_speed
                + self.offset_lin_forward_damping_speed
            )
        )
        # print("lin_damp: ", lin_damp)
        quad_damp = (
            (self.quadratic_damping + self.offset_nonlin_damping).mT * torch.abs(vel.mT)
        ).mT
        # print("quad_damp: ", quad_damp)
        # scaling and adding both matrices
        damping_matrix = (lin_damp + quad_damp) * self.scaling_damping
        # print("damping_matrix: ", damping_matrix)
        return damping_matrix

    """ 
    def GetAddedMass(self):
        print(torch.tensor(self.scaling_added_mass * (self.added_mass + self.offset_added_mass), device=self.device))
        return torch.tensor(self.scaling_added_mass * (self.added_mass + self.offset_added_mass), device=self.device)
    

    #negligeable in our case
    def ComputeAddedCoriolisMatrix(self, vel):

        // This corresponds to eq. 6.43 on p. 120 in
        // Fossen, Thor, "Handbook of Marine Craft and Hydrodynamics and Motion
        // Control", 2011  
        
        ##all is zero for now 

        ab = torch.matmul(self.GetAddedMass().mT, vel).mT  #num envs * 6
        Sa = -1 * torch.cross(torch.zeros([self._num_envs,6],device=self.device),torch.transpose(ab[:,:3],0,1), dim=1)
        self._Ca[-3:,:3] = Sa
        self._Ca[:3,-3:] = Sa
        self._Ca[-3:,-3:] = -1 * torch.cross(torch.zeros([3,self._num_envs]),ab[:,-3:].mT, dim=1) 
        
        return 
    
    
    def ComputeAcc(self, velRel, time, alpha):
    #Compute Fossen's nu-dot numerically. This is mandatory as Isaac does
    #not report accelerations

        if self._last_time < 0:
            self._last_time = time
            self._last_vel_rel = velRel
            return

        dt = time #time - self._last_time
        if dt <= 0.0:
            return

        acc = (velRel - self._last_vel_rel) / dt

        #   TODO  We only have access to the acceleration of the previous simulation
        #       step. The added mass will induce a strong force/torque counteracting
        #       it in the current simulation step. This can lead to an oscillating
        #       system.
        #       The most accurate solution would probably be to first compute the
        #       latest acceleration without added mass and then use this to compute
        #       added mass effects. This is not how gazebo works, though.

        self._filtered_acc = (1.0 - alpha) * self._filtered_acc + alpha * acc
        self._last_time = time
        self._last_vel_rel = velRel.copy()

        """

    def ComputeHydrodynamicsEffects(
        self, time, quaternions, world_vel, use_water_current, flow_vel
    ):
        rot_mat = pytorch3d.transforms.quaternion_to_matrix(quaternions)
        rot_mat_inv = rot_mat.mT

        self.local_lin_velocities = getLocalLinearVelocities(
            world_vel[:, :3], rot_mat_inv
        )
        self.local_ang_velocities = getLocalAngularVelocities(
            world_vel[:, 3:], rot_mat_inv
        )

        self.local_velocities = torch.hstack(
            [self.local_lin_velocities, self.local_ang_velocities]
        )

        if use_water_current:
            flow_vel = torch.tensor(flow_vel, device=self.device)

            if flow_vel.dim() == 1:
                flow_vel = flow_vel.unsqueeze(0).expand_as(world_vel[:, :3])

            self.local_flow_vel = getLocalLinearVelocities(flow_vel, rot_mat_inv)
            self.relative_lin_velocities = (
                getLocalLinearVelocities(world_vel[:, :3], rot_mat_inv)
                - self.local_flow_vel
            )
            self.local_velocities = torch.hstack(
                [self.relative_lin_velocities, self.local_ang_velocities]
            )

        # Update added Coriolis matrix
        # self.ComputeAddedCoriolisMatrix(self.local_velocities)
        # Update damping matrix
        damping_matrix = self.ComputeDampingMatrix(self.local_velocities)
        # Filter acceleration (see issue explanation above)
        # self.ComputeAcc(self.local_velocities, time, self.alpha)
        # We can now compute the additional forces/torques due to this dynamic
        # effects based on Eq. 8.136 on p.222 of Fossen: Handbook of Marine Craft ...
        # Damping forces and torques
        self.drag = -1 * damping_matrix * self.local_velocities
        # Added-mass forces and torques
        # added = torch.matmul(-self.GetAddedMass(), self._filtered_acc)
        # reshaped_added_tensor = torch.cat((added, torch.zeros(3 * 6 - len(added))), dim=0).view(3, 6)

        # Added Coriolis term
        # cor = torch.matmul(-self._Ca, self.local_velocities.mT).mT

        # All additional (compared to standard rigid body) Fossen terms combined.

        # cor and added should be zero from now

        # print("damping: ", damping)
        # print("added: ", reshaped_added_tensor)
        # print("cor: ", cor)

        # tau = damping + reshaped_added_tensor + cor

        # print("tau: ", tau)
        return self.drag
