 
function init( model, par )
	-- get the 'target_body' parameter from ScriptController, or set to "pelvis"
	-- target_body = model:find_body( scone.target_body or "pelvis" )
	calcn_r = model:find_body( "calcn_r" )
	calcn_l = model:find_body( "calcn_l" )

	-- target_actuator = model:find_actuator("motor_act")
	-- target_actuator = model:find_dof("knee_angle_r")
	
	-- hamstrings_r

	
	-- target_actuator = model:find_actuator(scone.target_act)
	target_actuator = model:find_dof(scone.target_act)
	-- target_actuator = model:find_dof("knee_angle_r")
	scone.debug("scone.target_act : " .. scone.target_act)
	scone.debug("target_actuator : " .. tostring(target_actuator))

	scone.debug("\nActuators:")
	for i = 1, model:actuator_count(), 1 do
		scone.debug(model:actuator(i):name())
	end
		
	peak_time = par:create_from_string( "peak_time", scone.peak_time )
	rise_time = par:create_from_string( "rise_time", scone.rise_time )
	duration_time = par:create_from_string( "duration_time", scone.duration_time )
	fall_time = par:create_from_string( "fall_time", scone.fall_time )
	peak_torque = par:create_from_string( "peak_torque", scone.peak_torque )
 
	-- initialize global variables that keep track of the device state
	device_startxxx = 0
	device_end = peak_time + duration_time + fall_time
	device_moment = 0
end
 
function update( model )
	

	local t = model:time() - device_startxxx
	local power_on = false

	if (t < 0) then
		device_startxxx = 0
		t = model:time() - device_startxxx
	end
	
	if t < peak_time - rise_time then
		device_moment = 0
	elseif (peak_time - rise_time <= t and t < peak_time) then
		device_moment = (peak_torque / 2) * (1 - math.cos(math.pi * ((t - (peak_time - rise_time))/rise_time)))
	elseif (peak_time <= t and t < peak_time + duration_time) then
		device_moment = peak_torque
	elseif ((peak_time + duration_time) <= t and t < (peak_time + duration_time + fall_time)) then
		device_moment = (peak_torque / 2) * (1 + math.cos(math.pi * ((t - (peak_time + duration_time))/fall_time)))
	elseif (peak_time + duration_time + fall_time) <= t then
		device_moment = 0
	end

	-- scone.debug( 
	-- 	model:time() - device_startxxx .. " , " .. model:time() .. " , " .. device_startxxx .. " , " .. tostring(calcn_l:contact_force().y) .. " , " .. tostring(calcn_r:contact_force().y))

	-- scone.debug( 
	-- 	t .. " , " .. device_end .. " , " .. model:time() .. " , " .. device_startxxx .. " , " .. tostring(calcn_l:contact_force().y) .. " , " .. tostring(calcn_r:contact_force().y))

	-- if (scone.target_act == "hip_flexion_r") then
	-- 	diff = calcn_r:contact_force().y - calcn_l:contact_force().y
	-- elseif (scone.target_act == "hip_flexion_l") then
	-- 	diff = calcn_l:contact_force().y - calcn_r:contact_force().y
	-- else
	-- 	scone.debug("error")
	-- end
	if (scone.target_act == "hip_flexion_r") then
		power_on = (target_actuator:position() > -0.1) and (target_actuator:velocity() > 0) and (calcn_r:contact_force().y < 1)
	elseif (scone.target_act == "hip_flexion_l") then
		power_on = (target_actuator:position() > -0.1) and (target_actuator:velocity() > 0) and (calcn_l:contact_force().y < 1)
	end

	
	-- scone.debug("scone.target_act : " .. scone.target_act .. " , muscle_moment: " .. target_actuator:position())

	target_actuator:add_input( 1.0 * device_moment )
	-- scone.debug("scone.target_act : " .. scone.target_act .. " , actuator_input: " .. target_actuator:input() .. " , " .. device_moment)
	if (power_on) and (t > device_end) then
		device_startxxx = model:time()
		-- scone.debug("device_start: " .. device_startxxx)
		-- scone.debug(" .................................................. ")
		device_end = peak_time + duration_time + fall_time
		power_on = false
	end
	-- return false to keep going
	return false;
end
