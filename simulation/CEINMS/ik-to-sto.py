import opensim as osim

# 加載模型
model = osim.Model("path_to_your_model.osim")

# 創建逆向運動學工具並設置參數
ik_tool = osim.InverseKinematicsTool("path_to_ik_setup_file.xml")
ik_tool.setModel(model)
ik_tool.setMarkerDataFileName("path_to_marker_data.trc")
ik_tool.setOutputMotionFileName("ik_output.mot")
ik_tool.run()

# 初始化系統
state = model.initSystem()

# 創建 Kinematics 分析工具並添加到模型中
kinematics = osim.Kinematics()
kinematics.setInDegrees(False)  # 設置為以弧度顯示
model.addAnalysis(kinematics)

# 初始化狀態並執行模擬
state = model.initSystem()

# 創建模擬管理器
manager = osim.Manager(model)
manager.initialize(state)

# 進行模擬並保存結果
manager.integrate(10.0)  # 模擬 10 秒

kinematics.getPositionStorage().printToXML("walk_subject01_Kinematics_q.sto")