 
function init( model, par )
 scone.info('init')
--scone.info( 'model : '.. tostring(model) )
--scone.info( 'par : '.. tostring(par) )

	calcn_r = model:find_body( "calcn_r" )
	calcn_l = model:find_body( "calcn_l" )

	-- target_actuator = model:find_actuator("motor_act")
	-- target_actuator = model:find_dof("knee_angle_r")
	
	target_actuator = model:find_dof("knee_angle_r")

	 scone.debug("\nActuators:")
	 for i = 1, model:actuator_count(), 1 do
	 	scone.debug(model:actuator(i):name())
	 end

end
 
function update( model )
	local t = model:time()
	scone.info(tostring(t))



	return false;
end

function store_data(frame)

end
