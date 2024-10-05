 
function init( model, par )
	-- get the 'target_body' parameter from ScriptController, or set to "pelvis"
	-- target_body = model:find_body( scone.target_body or "pelvis" )
	calcn_r = model:find_body( "calcn_r" )
	calcn_l = model:find_body( "calcn_l" )

	-- target_actuator = model:find_actuator("motor_act")
	-- target_actuator = model:find_dof("knee_angle_r")
	
	target_actuator = model:find_dof("knee_angle_r")

	-- scone.debug("\nActuators:")
	-- for i = 1, model:actuator_count(), 1 do
	-- 	scone.debug(model:actuator(i):name())
	-- end
		
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

	-- scone.debug( model:time() - device_startxxx .. " , " .. model:time() .. " , " .. device_startxxx .. " , " .. tostring(calcn_l:contact_force().y) .. " , " .. tostring(calcn_r:contact_force().y))

	target_actuator:add_input( 1.0 * device_moment )
	if (calcn_r:contact_force().y - calcn_l:contact_force().y > 0) and (t > device_end) then
		device_startxxx = model:time()
		scone.debug("device_start: " .. device_startxxx)
		device_end = peak_time + duration_time + fall_time
	end
	-- return false to keep going
	return false;
end
