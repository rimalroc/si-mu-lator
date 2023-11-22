import numpy as np
from matplotlib import pyplot as plt
import os
import contextlib
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error

norm_a = 0.43
norm_x = 0.102
mult_fact = 20
mult_facta = 0.1167

def print_dict(d, indent=0):
    align=20
    for key, value in d.items():
        print('  ' * indent + str(key), end='')
        if isinstance(value, dict):
            print()
            print_dict(value, indent+1)
        else:
            print(':' + ' ' * (20 - len(key) - 2 * indent) + str(value))
            
def histo (data,layer):
    h, b = np.histogram(data, bins=100)
    plt.figure(figsize=(7,7))
    plt.bar(b[:-1], h, width=b[1]-b[0])
    plt.semilogy()
    plt.suptitle(f'{layer}')
    print(layer)
    print('% of zeros = {}, min: {}, max: {}'.format(np.sum(data==0)/np.size(data)*100, b[0], b[-1]))

    
def plot_performance(model_dict, plot_residuals=False,
                     ranges=[10,20],PLOT_HLS = True, save=False, card='atlas_nsw_pad_z0'):
    mod = model_dict['name']
    print(f"# INFO: plotting {mod}")
    x_keras = model_dict['x_keras']
    a_keras = model_dict['a_keras']
    x_hls = model_dict['x_hls']
    a_hls = model_dict['a_hls']
    if plot_residuals:
        x_keras = x_keras-x_true
        a_keras = a_keras-a_true
        x_hls = x_hls-x_true
        a_hls = a_hls-a_true
    ncols, nrows = (2,1)
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(6*ncols,4*nrows))
    axs = axs.flatten()
    fit_x_std = (lfits[this_cut][:,1] - data['ev_mu_x'][this_cut])[:len(a_true)].std()
    axs[0].hist((lfits[this_cut][:,1] - data['ev_mu_x'][this_cut])[:len(a_true)], histtype='step', range=(-ranges[0],ranges[0]), color='C0',density=1,
                    bins=100, label=r'$\chi^2$: $\mu \approx 0$ ' +f'std={(fit_x_std/norm_x):.2f}u')
    fit_mu_x_std = (lfits_mu[this_cut][:,1] - data['ev_mu_x'][this_cut])[:len(a_true)].std()
    axs[0].hist((lfits_mu[this_cut][:,1] - data['ev_mu_x'][this_cut])[:len(a_true)], histtype='step', range=(-ranges[0],ranges[0]), color='C1',density=1,
                    bins=100, label=r'$\chi^2_{\mu}$: $\mu \approx 0$ ' +f'std={(fit_mu_x_std/norm_x):.2f}u')
    #print(f"mse x: {mean_squared_error(x_true[this_cut], x_hls[this_cut])}")
    print(f"mse x: {mean_squared_error(x_true, x_keras)}")
    x_std_keras = ((x_keras)).std()
    x_mean_keras = ((x_keras)).mean()
    axs[0].hist((x_keras), histtype='step', range=(-ranges[0],ranges[0]),
                bins=100, color='red', density=1,
                        label=r'$NN$: $\mu$=' + f'{(x_mean_keras/norm_x):.2f}u std={(x_std_keras/norm_x):.2f}u', linewidth=2)
    if PLOT_HLS:
        x_std_hls = ((x_hls)).std()
        x_mean_hls = ((x_hls)).mean()
        axs[0].hist((x_hls), histtype='step', range=(-ranges[0],ranges[0]),
                bins=100, color='blue', density=1,
                        label=r'$NN_{hls}$: $\mu$=' + f'{(x_mean_hls/norm_x):.2f}u std={(x_std_hls/norm_x):.2f}u', linewidth=2)
    #    axs[0].annotate(f'int. bit: {int_bit}\nfra. bit: {fra_bit}', xy=(0, 0), xytext=(0.7, 0.7),
    #                ha='left', va='bottom', xycoords='axes fraction', textcoords= 'axes fraction',#
    #                    size=15, color='blue', style="normal")

    fit_theta_std = (lfits_theta[this_cut] - data['ev_mu_theta'][Y_mu==1]*1000).std()
    axs[1].hist((lfits_theta[this_cut] - data['ev_mu_theta'][Y_mu==1]*1000)[:len(a_true)], histtype='step', range=(-ranges[1],ranges[1]), density=1, color='C0',
                    bins=100, label=r'$\chi^2$: $\mu \approx 0$ ' +f'std={(fit_theta_std/norm_a):.2f}u')
                    #                bins=50, label=f'Chi2 ( {norm_fit_chi2[0]:.4f}, {norm_fit_chi2[1]:.4f} ) {std_fit}')
    fit_thetha_mu_std = (lfits_mu_theta[this_cut] - data['ev_mu_theta'][Y_mu==1]*1000).std()
    axs[1].hist((lfits_mu_theta[this_cut] - data['ev_mu_theta'][Y_mu==1]*1000)[:len(a_true)], histtype='step', range=(-ranges[1],ranges[1]), density=1, color='C1',
                    bins=100, label=r'$\chi^2_{\mu}$: $\mu \approx 0$ ' +f'std={(fit_thetha_mu_std/norm_a):.2f}u')
    #print(f"mse a: {mean_squared_error(a_true[this_cut], a_hls[this_cut]*mult_facta*1000)}")
    a_std_keras = ((a_keras)).std()
    a_mean_keras = ((a_keras)).mean()
    axs[1].hist(( a_keras), 
                        histtype='step', range=(-ranges[1],ranges[1]),
                bins=100, color='red', density=1,
                        label=r'$NN$: $\mu$=' + f'{(a_mean_keras/norm_a):.2f}u std={(a_std_keras/norm_a):.2f}u', linewidth=2)
    if PLOT_HLS:
        a_std_hls = ((a_hls)).std()
        a_mean_hls = ((a_hls)).mean()
        axs[1].hist((a_hls), 
                        histtype='step', range=(-ranges[1],ranges[1]),
                bins=100, color='blue', density=1,
                        label=r'$NN_{hls}$: $\mu$=' + f'{(a_mean_hls/norm_a):.2f}u std={(a_std_hls/norm_a):.2f}u', linewidth=2)
    #    axs[1].annotate(f'int. bit: {int_bit}\nfra. bit: {fra_bit}', xy=(0, 0), xytext=(0.7, 0.7),
    #                ha='left', va='bottom', xycoords='axes fraction', textcoords= 'axes fraction',
    #                    size=15, color='blue', style="normal")

    axs[0].set_xlabel('Pred X - True X [mm]')
    axs[1].set_xlabel(r'Pred $\theta$ - True $\theta$ [mrad]')
    axs[0].legend(ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.1))
    axs[1].legend(ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.1))
    axs[0].set_ylim([1e-4, 10])
    axs[1].set_ylim([1e-4, 10])
    axs[0].set_yscale('log')
    axs[1].set_yscale('log')
    if plot_residuals: plt.suptitle(f'mod {mod} {card} residual')
    else: plt.suptitle(f'mod {mod} {card}')
    if save:
        plt.savefig(f'model_plots/hls{PLOT_HLS}_{mod}_{card}.pdf')
        plt.savefig(f'model_plots/hls{PLOT_HLS}_{mod}_{card}.png')

        
import hls4ml

from qkeras.utils import _add_supported_quantized_objects,load_qmodel, quantized_model_debug
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from tensorflow_model_optimization.sparsity.keras import strip_pruning
from os.path import exists
co = {}
_add_supported_quantized_objects(co)
co['PruneLowMagnitude'] = pruning_wrapper.PruneLowMagnitude

#REWRITE_CONF=False
#model_id = 9
#mod = models[model_id]['name']
#qmodel = load_qmodel(mod_dir+models[model_id]['dir'] + '/weights.h5',compile=False, custom_objects=co)
#qmodel = strip_pruning(qmodel)
#help(qmodel)
def conv_to_hls(model_dict, qmodel, default_precision="26,6",default_reuse_factor=1, REWRITE_CONF=False, verbose=False, card='atlas_nsw_pad_z0'):
    
    for layer in qmodel.get_config()['layers']:
        for key in layer.keys():
            if 'class_name' in key:
                if 'PruneLowMagnitude' in layer[key]:
                    print("model includes prunning, removing..")
                    qmodel = strip_pruning(qmodel)
                    break
        else:
            continue
        break

#    precision=default_precision#str(int_bit + fra_bit) + ',' + str(int_bit)
    print(f"precision: {default_precision} model: {model_dict['dir']}")
    hls4ml.model.optimizer.OutputRoundingSaturationMode.layers = ['Activation']
    hls4ml.model.optimizer.OutputRoundingSaturationMode.rounding_mode = 'AP_RND'
    hls4ml.model.optimizer.OutputRoundingSaturationMode.saturation_mode = 'AP_SAT'
    hls4ml_conf_file = model_dict['dir'] + '/hls4ml_conf.npy'
    FILE_EXISTS = exists(hls4ml_conf_file)

    if REWRITE_CONF or FILE_EXISTS == False:
        print("# INFO: Rewrite configuration of layers")
        config = hls4ml.utils.config_from_keras_model(qmodel, granularity='name', 
                                                      default_precision=f'ap_fixed<{default_precision}>',
                                                      default_reuse_factor=default_reuse_factor)
        t_size = max( int(2**(1+float(default_precision.split(',')[1]))  ), 1024)
        config['Model']['Precision'] = f'ap_fixed<{default_precision}>'
        for layer in config['LayerName'].keys():
            print(layer)
            if 'softmax' in layer or 'sigmoid' in layer or 'relu' in layer or 'tanh' in layer:
                config['LayerName'][layer]['table_t'] = f'ap_ufixed<16,0>'
        #        config['LayerName'][layer]['Precision']['result'] = f'ap_ufixed<16,0>'
        #        config['LayerName'][layer]['table_size'] = f'{t_size}'
            config['LayerName'][layer]['Trace'] = True
#            if 'C1D' in layer and 'relu' not in layer:# or 'output' in layer:
#                config['LayerName'][layer]['Precision']['weight'] = f'ap_fixed<20,1>' #check the original values by commenting
#                config['LayerName'][layer]['Precision']['bias' ] = f'ap_fixed<14,2>' #check the original values by commenting
#                config['LayerName'][layer]['Precision']['result'] = f'ap_fixed<28,3>'
#            if 'F_dense' in layer and 'relu' not in layer:
 #               config['LayerName'][layer]['Precision']['result'] = f'ap_fixed<24,5>'
#                config['LayerName'][layer]['Precision']['weight'] = f'ap_fixed<18,2>'
#                config['LayerName'][layer]['Precision']['bias'] = f'ap_fixed<18,2>'
            if 'normalization' in layer:
        #        config['LayerName'][layer]['Precision']['mean'] = f'ap_fixed<7,6>' #check the original values by commenting
        #        config['LayerName'][layer]['Precision']['variance'] = f'ap_fixed<8,7>' #check the original values by commenting
        #        config['LayerName'][layer]['Precision']['beta'] = f'ap_fixed<7,6>' #check the original values by commenting
        #        config['LayerName'][layer]['Precision']['gamma'] = f'ap_fixed<8,7>' #check the original values by commenting
                config['LayerName'][layer]['Precision']['scale'] = f'ap_fixed<25,8>' #check the original values by commenting
                config['LayerName'][layer]['Precision']['bias' ] = f'ap_fixed<25,7>' #check the original values by commenting
        #        config['LayerName'][layer]['Precision']['result'] = f'ap_fixed<24,6>'
#            if 'output' in layer and 'linear' not in layer:
#                config['LayerName'][layer]['Precision']['weight'] = f'ap_fixed<18,2>' #check the original values by commenting
#                config['LayerName'][layer]['Precision']['bias' ] = f'ap_fixed<12,2>' #check the original values by commenting
 #               config['LayerName'][layer]['Precision']['result'] = f'ap_fixed<18,2>'

    #    conf_dicts[mod] = cfg
        np.save(hls4ml_conf_file, config)
        # If changing configuration, delete previous report and results
        with contextlib.suppress(FileNotFoundError):
            os.remove(model_dict['dir'] + '/hls4ml_report.npy')
        with contextlib.suppress(FileNotFoundError):
            os.remove(model_dict['dir'] + f'/hls4ml_results_{card}.npy')
    else:
        print("# INFO: Loading saved configuration of layers")
        config = np.load(hls4ml_conf_file,allow_pickle=True).item()
    #    cfg = conf_dicts[mod]
    out_loc_name = "./"
    proj_loc = out_loc_name + f'./myproject_prj/'
    fpgapart = 'xcvu13p-fsga2577-2-e'
    cfg = hls4ml.converters.create_config(backend='Vivado')
    cfg['IOType']     = 'io_stream' # Must set this if using CNNs!
    cfg['HLSConfig']  = config
    cfg['KerasModel'] = qmodel
    cfg['OutputDir']  = proj_loc
    cfg['XilinxPart'] = fpgapart
    model_dict['config'] = config
    if verbose: helper.print_dict(cfg)
    hls_model = hls4ml.converters.keras_to_hls(cfg)
    return hls_model
    
#hls4ml.model.profiling.numerical(model=qmodel, hls_model=hls_model, X=x_test[:1000])

def compile_predict_profile(model_dict, hls_model,model, x_test, x_true, a_true, REWRITE_RESULTS=False, card='atlas_nsw_pad_z0'):
    hls4ml_results_file = model_dict['dir'] + f'/hls4ml_results_{card}.npy'
    FILE_EXISTS = exists(hls4ml_results_file)
    if FILE_EXISTS and REWRITE_RESULTS == False:
        print("# INFO: loading saved results")
        stored_dict = np.load(hls4ml_results_file,allow_pickle=True)
        model_dict.update(stored_dict.item())
    else:
        print(f"# INFO: predicting new results for card {card}")
        model_dict['card'] = card
        preds = model.predict(x_test)#, batch_size=1024)
#        preds = model.predict(X_prep[:,:,vars_of_interest][this_cut])
        model_dict['x_keras'] = preds[:,0]*mult_fact
        model_dict['a_keras'] = preds[:,2]*mult_facta*1000
#        print(0)
        hls_model.compile()
#        print(1)
        model_dict['y_hls'] = hls_model.predict(np.ascontiguousarray(x_test))
        model_dict['hls4ml_pred'], model_dict['hls4ml_trace'] = hls_model.trace(np.ascontiguousarray(x_test))
        model_dict['keras_trace'] = hls4ml.model.profiling.get_ymodel_keras(model, x_test)
#        print(2)
        model_dict['x_hls'] = model_dict['y_hls'][:,0]*mult_fact
        model_dict['a_hls'] = model_dict['y_hls'][:,2]*mult_facta*1000
        model_dict['x_std_hls'] = ((model_dict['x_hls']-x_true)).std()
        model_dict['x_mean_hls'] = ((model_dict['x_hls']-x_true)).mean()
        model_dict['a_std_hls'] = ((model_dict['a_hls']-a_true)).std()
        model_dict['a_mean_hls'] = ((model_dict['a_hls']-a_true)).mean()
        model_dict['x_std_keras'] = ((model_dict['x_keras']-x_true)).std()
        model_dict['x_mean_keras'] = ((model_dict['x_keras']-x_true)).mean()
        model_dict['a_std_keras'] = ((model_dict['a_keras']-a_true)).std()
        model_dict['a_mean_keras'] = ((model_dict['a_keras']-a_true)).mean()
        model_dict['x_mse_keras'] = mean_squared_error(x_true, model_dict['x_keras'])
        model_dict['a_mse_keras'] = mean_squared_error(a_true, model_dict['a_keras'])
        
        q16, q84 = np.percentile(((model_dict['x_keras']-x_true)), [16 ,84])
        model_dict['x_percentil'] = (q84 - q16)/2.
        q16, q84 = np.percentile(((model_dict['a_keras']-a_true)), [16 ,84])
        model_dict['a_percentil'] = (q84 - q16)/2.
#        print(3)
        print (model_dict['x_mse_keras'])
        np.save(hls4ml_results_file,model_dict)
#        print(4)




def diff_dataset_plot(data, labels, title, y_lim=[0,25]):
    x = np.arange(len(labels))  # the label locations
    width = 0.2 # the width of the bars

    fig, ax = plt.subplots(figsize=(10,8))
    rects = []

    for i, (key, values) in enumerate(data.items()):
        # draw the bars
        rects.append(ax.bar(x + i*width, values, width, label=key))#, color=colors[i]))


    # add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('std [units]', fontsize=18)
    ax.set_xlabel('Test Datasets', fontsize=18)
    ax.set_title(title, fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    #ax.legend()
    #ax.legend(ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.1))
    ax.set_ylim(y_lim)
    # modify legend properties
    legend = ax.legend(ncol=2, loc='upper right', fontsize=16)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(1)
    legend.get_frame().set_alpha(1)

    # control the layout of the legend
    legend._legend_box.align = "center" 
    legend._set_loc(2)
    legend._set_loc((0.02, 0.85))


    def autolabel(rects):
        # function to attach a text label to each rectangl
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:0.2f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height), 
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=14)

    # add the value label
    for bar_group in rects:
        autolabel(bar_group)


    plt.savefig(f"model_plots/diff_dataset_results_{title.split(' ')[0:3]}.pdf")
    plt.show()