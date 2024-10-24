import opensim as osim

path_to_model = "C:/Users/sean9/Desktop/NTK_Cap/Model_Pose2Sim_Halpe26.osim"
# path_to_scaling_setup_file = ""
path_to_marker_data = "C:/Users/sean9/Desktop/NTK_Cap/test_data/path1_03/Empty_project_filt_0-30.trc"
path_to_ik_output = "C:/Users/sean9/Desktop/NTK_Cap/output/ik_output.mot"
path_to_kinematics_q = "C:/Users/sean9/Desktop/NTK_Cap/output/walk_subject01_Kinematics_q.sto"

# scale_tool = osim.ScaleTool()
# scale_tool.setSetupFileName(path_to_scaling_setup_file)
# scale_tool.getGenericModelMaker().setModelFileName("path_to_generic_model.osim")
# scale_tool.getModelScaler().setMarkerFileName("path_to_marker_data.trc")
# scale_tool.getModelScaler().setOutputModelFileName("scaled_model.osim")
model = osim.Model(path_to_model)
ik_tool = osim.InverseKinematicsTool()
ik_tool.setModel(model)
ik_tool.setMarkerDataFileName(path_to_marker_data)
ik_tool.setOutputMotionFileName(path_to_ik_output)
ik_tool.run()
state = model.initSystem()

kinematics = osim.Kinematics()
kinematics.setInDegrees(False)
model.addAnalysis(kinematics)
state = model.initSystem()
manager = osim.Manager(model)
manager.initialize(state)
manager.integrate(10.0)

kinematics.getPositionStorage().printToXML(path_to_kinematics_q)
