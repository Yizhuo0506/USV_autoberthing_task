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
        metacentric_width,
        metacentric_length,
        average_buoyancy_force_value,
        amplify_torque,
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

        # Buoyancy
        self.water_density = water_density
        self.gravity = gravity
        self.metacentric_width = metacentric_width
        self.metacentric_length = metacentric_length
        self.archimedes_force_global = torch.zeros(
            (self._num_envs, 3), dtype=torch.float32, device=self.device
        )
        self.archimedes_torque_global = torch.zeros(
            (self._num_envs, 3), dtype=torch.float32, device=self.device
        )
        self.archimedes_force_local = torch.zeros(
            (self._num_envs, 3), dtype=torch.float32, device=self.device
        )
        self.archimedes_torque_local = torch.zeros(
            (self._num_envs, 3), dtype=torch.float32, device=self.device
        )

        # data
        self.average_buoyancy_force_value = average_buoyancy_force_value
        self.amplify_torque = amplify_torque

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

    def compute_squared_drag(self, boat_velocities, quaternions):
        """this function implements the drag, rotation drag is needed becaurle of where archimedes is applied. if the boat start to rate around x for
        example, since archimedes is applied onto the center, isaac sim will believe that the boat is still under water and so the boat is free to rotate around and
        y. So to prevent this behaviour, if we don't want to create 4 rigid bodies as talked above, we are forced to add a drag + stabilizer to the simulation.

        coefficients  = 0.5 * ρ * v^2 * A * Cd

        ρ (rho) is the density of the surrounding fluid in kg/m³.
        v is the velocity of the object relative to the fluid in m/s.
        A is the reference area of the object perpendicular to the direction of motion in m². This is usually the frontal area of the object exposed to the fluid flow.
        Cd is the drag coefficient (dimensionless) that depends on the shape and roughness of the object. This coefficient is often determined experimentally.

        for our boxes, A ~ 0.2 , ρ ~ 1000, Cd ~ 0.05"""

        rot_mat = self.getWorldToLocalRotationMatrix(quaternions)
        rot_mat_inv = rot_mat.mT

        local_lin_velocities = getLocalLinearVelocities(
            boat_velocities[:, :3], rot_mat_inv
        )
        local_ang_velocities = getLocalAngularVelocities(
            boat_velocities[:, 3:], rot_mat_inv
        )

        self.drag[:, :3] = -(
            self.drag_coefficients[:, :3].mT
            * torch.abs(local_lin_velocities).mT
            * local_lin_velocities.mT
        ).mT
        self.drag[:, 3:] = -(
            self.drag_coefficients[:, 3:].mT
            * torch.abs(local_ang_velocities).mT
            * local_ang_velocities.mT
        ).mT
        # print ("drag: ", self.drag)

        return self.drag

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

    def ComputeHydrodynamicsEffects(self, time, quaternions, world_vel):
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

    def compute_archimedes_metacentric_global(self, submerged_volume, rpy):
        """This function apply the archimedes force to the center of the boat"""

        """Ideally, this function should not be applied only at the center of the boat, but at the center of the volume submerged underwater.
        In this case, if the boat start to rotate around x and y axis, the part of the boat who isn't underwater anymore have no force except gravity applied,
        it automatically balance the boat. But that would require to create 4 rigid body at each corner of the boat and then check which one of them is underwater.
        """

        roll, pitch = rpy[:, 0], rpy[:, 1]  # roll and pich are given in global frame

        # compute buoyancy force
        self.archimedes_force_global[:, 2] = (
            -self.water_density * self.gravity * submerged_volume
        )  # buoyancy force

        # torques expressed in global frame, size is (num_envs,3)
        self.archimedes_torque_global[:, 0] = (
            -1
            * self.metacentric_width
            * (torch.sin(roll) * self.archimedes_force_global[:, 2])
        )
        self.archimedes_torque_global[:, 1] = (
            -1
            * self.metacentric_length
            * (torch.sin(pitch) * self.archimedes_force_global[:, 2])
        )

        self.archimedes_torque_global[:, 0] = (
            -1
            * self.metacentric_width
            * (torch.sin(roll) * self.average_buoyancy_force_value)
        )  # cannot multiply by the buoyancy force in isaac sim because of the simulation rate (low then high value)
        self.archimedes_torque_global[:, 1] = (
            -1
            * self.metacentric_length
            * (torch.sin(pitch) * self.average_buoyancy_force_value)
        )

        # debugging
        # print("self.archimedes_force global: ", self.archimedes_force_global[0,:])
        # print("self.archimedes_torque global: ", self.archimedes_torque_global[0,:])

        return self.archimedes_force_global, self.archimedes_torque_global

    def compute_archimedes_metacentric_local(self, submerged_volume, rpy, quaternions):
        """This function apply the archimedes force to the center of the boat"""

        """Ideally, this function should not be applied only at the center of the boat, but at the center of the volume submerged underwater.
        In this case, if the boat start to rotate around x and y axis, the part of the boat who isn't underwater anymore have no force except gravity applied,
        it automatically balance the boat. But that would require to create 4 rigid body at each corner of the boat and then check which one of them is underwater.
        """

        # get archimedes global force
        self.compute_archimedes_metacentric_global(submerged_volume, rpy)

        # get rotation matrix from quaternions in world frame, size is (3*num_envs, 3)
        R = getWorldToLocalRotationMatrix(quaternions)

        # print("R:", R[0,:,:])

        # Arobot = Rworld * Aworld. Resulting matrix should be size (3*num_envs, 3) * (num_envs,3) =(num_envs,3)
        self.archimedes_force_local = torch.bmm(
            R.mT, torch.unsqueeze(self.archimedes_force_global, 1).mT
        )  # add batch dimension to tensor and transpose it
        self.archimedes_force_local = self.archimedes_force_local.mT.squeeze(
            1
        )  # remove batch dimension to tensor

        self.archimedes_torque_local = torch.bmm(
            R.mT, torch.unsqueeze(self.archimedes_torque_global, 1).mT
        )
        self.archimedes_torque_local = self.archimedes_torque_local.mT.squeeze(1)

        # not sure if torque have to be multiply by the rotation matrix also.
        self.archimedes_torque_local = self.archimedes_torque_global

        return torch.hstack(
            [
                self.archimedes_force_local,
                self.archimedes_torque_local * self.amplify_torque,
            ]
        )

    # alternative of archimedes torque
    """
    def stabilize_boat(self,yaws):
        # Roll Stabilizing Force = -k_roll * θ_x, Yaw Stabilizing Force = -k_yaw * θ_z 

        K=5.0 #by hand
        force=torch.zeros((self._num_envs, 3), dtype=torch.float32)
        
        force[:,0] = - K * yaws[:,0]
        force[:,1] = - K * yaws[:,1]

        return force
    """
