from metrics5 import Metrics

phantoms = [9, 13,16,19,33,36,37]  #[35,27,15,3,39,7,33]
architecture='unet'
pos=1
it=2 #7,8,10
# mode='horizontal'
dilation_f=4
width=1
epochs=10
depth=80
load_model: True
loss_f: "L2"
run_folder= '{}_pos{}_width{}_depth{}_dil{}_ep{}_it{}/'.format(architecture, pos, width, depth, dilation_f,epochs, it)


cal= Metrics(phantoms,  architecture, pos, [it], depth, width, dilation_f, run_folder)
print(cal.calculate_metrics())

