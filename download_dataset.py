from roboflow import Roboflow

rf = Roboflow(api_key="RjPwGpeGv84rwk7b5xnf")

project = rf.workspace("joseph-nelson").project("hard-hat-sample")
dataset = project.version(1).download("yolov8", location="datasets/ppe")

print("Done!")