import torch

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils import get_inverse_normalization


# Enable for color pairs
# colors = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),  
#          (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),  
#          (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),  
#          (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),  
#          (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229),
          
#          (255, 120, 86), (255, 180, 186), (132, 183, 1), (132, 183, 101),]

# # Rescale to values between 0 and 1 
# for i in range(len(colors)):  
#     r, g, b = colors[i]  
#     colors[i] = (r / 255., g / 255., b / 255.)

# Enable for single colors
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
          '#1f77f4', '#ff7ffe', '#2ca0fc', '#d627f8', '#9467fd', '#8c56fb', '#e377f2', '#7f7fff', '#bcbdf2', '#17beff']

def plot_curves_advanced(plot_data, plot_format, suptitle, margin=0.05, log=False, clean=True):
    r""""""
    
    # First we define the size of the plot depending on the number of plots
    columns = 2 if len(plot_data) > 1 else 1
    rows = int(np.ceil(len(plot_data) / columns) )
    figsize = (6*columns,rows*4)
    fig, ax = plt.subplots(rows, columns, figsize=figsize, squeeze=False)
    
    # First we loop over the datasets and labels
    for i, (sub_plot_data, sub_plot_format) in enumerate(zip(plot_data, plot_format)):

        idx = (int(i/columns), int(i%columns))  # plot idx
        xlabel, ylabel, zlabel, title, plot_range = sub_plot_format
            
        ax[idx].set_xlabel(xlabel, fontsize=17)
        ax[idx].set_ylabel(ylabel, fontsize=17)

        if not clean:
            ax[idx].set_title(title)
        
        ax[idx].minorticks_on
        ax[idx].grid(True, which='both')

        if log:
            ax[idx].yscale("log")
        
        # The plot range is optional and determins the limits of the plot
        if plot_range is not None:
            min_val_x, max_val_x, xstep = plot_range[0][0], plot_range[0][1], plot_range[0][2]
            min_val_y, max_val_y, ystep = plot_range[1][0], plot_range[1][1], plot_range[1][2]
            
            ax[idx].set_xlim([min_val_x - margin, max_val_x + margin])
            ax[idx].set_ylim([min_val_y - margin, max_val_y + margin])
            
            ax[idx].set_xticks(np.arange(min_val_x, max_val_x + xstep, xstep))
            ax[idx].set_yticks(np.arange(min_val_y, max_val_y + ystep, ystep))
            
        # We loop over all the data that is going in the subplot
        for j, data in enumerate(sub_plot_data):
            
            xdata, ydata, zdata, label = data

            ax[idx].plot(xdata, ydata, label=label, linestyle='solid', c=colors[j])
            
            if zdata is not None:
                plot = ax[idx].scatter(xdata, ydata, linestyle='--', s=100, c=zdata, cmap='Reds')
                divider = make_axes_locatable(ax[idx])
                cax = divider.append_axes('right', size='5%', pad=0.005)
                fig.colorbar(plot, cax=cax, orientation='vertical')

                cax.get_yaxis().labelpad = 15
                cax.set_ylabel(zlabel, rotation=270)
            
            ax[idx].legend()
    
    if not clean:
        fig.suptitle(suptitle, y=1, fontsize=20)

    fig.tight_layout()
        
    return fig
        
    
def plot_points(plot_data, plot_format, suptitle, margin=0.05, legend_loc='out', trendline=True, corr=True, columns=2,
               fragment=False, merge_title=False, center_ylabel=False):
    """
    datasets is of the form (#plots, #data_in_plots, 2, #num_points) 
    values is of the form (#plots, #data_in_plots, #vals) 
    labels are of the form (#plots, #data_in_plots) 
    xlabels, ylabels, titles are of the form (#plots, )
    """
    
    # First we define the size of the plot depending on the number of plots
    
    if not fragment:
        columns = columns if len(plot_data) > 1 else 1
        rows = int(np.ceil(len(plot_data) / columns) )
        figsize = (6*columns,rows*4)
        fig, ax = plt.subplots(rows, columns, figsize=figsize, squeeze=False)
    
    for i, (sub_plot_data, sub_plot_format) in enumerate(zip(plot_data, plot_format)):
        
        xlabel, ylabel, zlabel, title, plot_range = sub_plot_format
        
        if fragment:
            fig, ax = plt.subplots(1, 1, figsize=(6,4), squeeze=False)
            idx = (0, 0)
        else:
            idx = (int(i/columns), int(i%columns))  # plot idx
        
        if merge_title:
            ax[idx].text(0.05, 0.95, title, ha='left', va='top', transform=ax[idx].transAxes, size=20, alpha=0.8)
        else:
            ax[idx].set_title(title, fontsize=20)
            
        ax[idx].set_xlabel(xlabel, fontsize=17)
        ax[idx].set_ylabel(ylabel, fontsize=17)
        
        if center_ylabel:    
            ax[idx].yaxis.set_label_coords(-0.1,1.1)
            
        # The plot range is optional and determins the limits of the plot
        if plot_range is not None:
            min_val_x, max_val_x, xstep = plot_range[0][0], plot_range[0][1], plot_range[0][2]
            min_val_y, max_val_y, ystep = plot_range[1][0], plot_range[1][1], plot_range[1][2]
            
            ax[idx].set_xlim([min_val_x - margin, max_val_x + margin])
            ax[idx].set_ylim([min_val_y - margin, max_val_y + margin])
            
            ax[idx].set_xticks(np.arange(min_val_x, max_val_x + xstep, xstep))
            ax[idx].set_yticks(np.arange(min_val_y, max_val_y + ystep, ystep))
        
        # We loop over all the data that is going in the subplot
        all_xdata = []
        all_ydata = []
        for j, data in enumerate(sub_plot_data):
            
            if len(data) == 4:
                xdata, ydata, zdata, label = data
                marker, color, xdata_e, ydata_e, zdata_e = 'o', colors[j], None, None, None
            elif len(data) == 6:
                xdata, ydata, zdata, label, marker, color = data
                xdata_e, ydata_e, zdata_e = None, None, None
            else:
                xdata, ydata, zdata, xdata_e, ydata_e, zdata_e, label, marker, color = data
            
            ax[idx].scatter(xdata, ydata, label=label, color=color, marker=marker, s=60) 
            
            if xdata_e is not None:
                ax[idx].errorbar(xdata, ydata, xdata_e, linestyle='-', marker='^')
            if ydata_e is not None:
                ax[idx].errorbar(xdata, ydata, ydata_e, linestyle='-', color=color, elinewidth=1,
                                capsize=3)
            
            all_xdata.append(float(xdata))
            all_ydata.append(float(ydata))
            
        if trendline:
            z = np.polyfit(all_xdata, all_ydata, 1)
            p = np.poly1d(z)
            x_min, x_max = np.argmax(all_xdata), np.argmin(all_xdata)
            ax[idx].plot([all_xdata[x_min], all_xdata[x_max]], [p(all_xdata)[x_min], p(all_xdata)[x_max]], "r--", 
                         alpha=0.5) # label='trendline'
        
        if corr:
            corr_coeff = np.round(np.corrcoef(all_xdata, all_ydata)[0][1], 3)
            
            if merge_title:
                ax[idx].text(0.05, 0.85, 'r = ' + str(corr_coeff), ha='left', va='top', transform=ax[idx].transAxes, size=14)
            else:
                ax[idx].text(0.05, 0.95, 'r = ' + str(corr_coeff), ha='left', va='top', transform=ax[idx].transAxes, size=14)

    if not fragment:
        handles, labels = ax[idx].get_legend_handles_labels()
        
        # bbox_to_anchor=(1, 0.99), bbox_to_anchor=(1.527, 0.965)
        # bbox_to_anchor=(1, 0.99), bbox_to_anchor=(1.265, 0.977)
        
        if legend_loc == 'out':
            fig.legend(handles[::2], labels[::2], bbox_to_anchor=(1, 0.99), ncol=1, labelspacing=3, framealpha=0,
                      mode='expand', fontsize=11) 
            fig.legend(handles[1::2], labels[1::2], bbox_to_anchor=(1.265, 0.977), ncol=1, labelspacing=3, framealpha=0,
                      fontsize=11) 
        else:
            fig.legend(handles, labels, loc='center right')

        fig.suptitle(suptitle, fontsize=20, y=1)
        fig.tight_layout()
    
    return fig



def plot_hist_advanced(plot_data, plot_format, suptitle='test', columns=2, density=True, clean=True):
    r"""
    datasets and labels are of the form (#plots, #data_in_plots, #num_points)
    xlabels and ylabels are of the form (#plots, )
    """
    
    # First we define the size of the plot depending on the number of plots
    columns = columns if len(plot_data) > 1 else 1
    rows = int(np.ceil(len(plot_data) / columns) )
    figsize = (6*columns,rows*4)
    fig, ax = plt.subplots(rows, columns, figsize=figsize, squeeze=False)

    # Then we loop over the datasets and the plot formatting
    for i, (sub_plot_data, sub_plot_format) in enumerate(zip(plot_data, plot_format)):
        
        idx = (int(i/columns), int(i%columns))
        xlabel, ylabel, title, bins, confidence = sub_plot_format
        
        # We get the maximimum value from the confidence interval of the joint data
        max_ci = get_max_ci(sub_plot_data, confidence=confidence)
        total_samples, samples_plotted = 0, 0
        
        # We then loop over all the data in the current plot
        for j, (data, label) in enumerate(sub_plot_data):
            
            data_sub = data[data <= max_ci]
            alpha = 1 / (j+1)

            if len(data_sub) > 0:
                ax[idx].hist(data_sub, bins=bins, label=label, density=density, alpha=alpha)
                # ax[idx].hist(data_sub, bins=bins, label=label, density=False, alpha=alpha, weights=np.ones(len(data_sub)) / len(data_sub))

            total_samples += len(data)
            samples_plotted += len(data_sub)

        ax[idx].set_xlabel(xlabel, fontsize=15)
        ax[idx].set_ylabel(ylabel, fontsize=15)
        ax[idx].spines['top'].set_visible(False)
        ax[idx].spines['right'].set_visible(False)

        if not clean:
            ax[idx].set_title(title + f' (plotted {np.round(samples_plotted/total_samples,3)}%)')  # ax[idx].text(0,10, f' plotted {np.round(samples_plotted/total_samples,3)}% of the data', fontsize=10)

        ax[idx].legend()
    
    if not clean:
        fig.suptitle(suptitle, size=15)

    fig.tight_layout()

    return fig

def get_max_ci(sub_plot_data, confidence=99):
    r"""
    Returns the highest value on the right of the max confidence intervals of all the given datasets.
    """
    
    # If the confidence is not in the specified confidence we return the confidence number
    if confidence in [99, 98, 95, 90, 80]:
        max_ci = 0
        for values, _ in sub_plot_data:
            if len(values) > 0:
                if confidence == 99:
                    ci = np.mean(values) + 2.58 * np.std(values)/1  #np.sqrt(len(dataset))
                elif confidence == 98:
                    ci = np.mean(values) + 2.33 * np.std(values)/1  #np.sqrt(len(dataset))
                elif confidence == 95:
                    ci = np.mean(values) + 1.96 * np.std(values)/1  #np.sqrt(len(dataset))
                elif confidence == 90:
                    ci = np.mean(values) + 1.645 * np.std(values)/1  #np.sqrt(len(dataset))
                elif confidence == 80:
                    ci = np.mean(values) + 1.28 * np.std(values)/1   #np.sqrt(len(dataset))
            else:
                ci = 0

            if ci > max_ci:  # the max confidence is the maximum value found on all confidence intervals
                max_ci = ci
            
        return max_ci
    
    return confidence   



def plot_bars_singlecolumn(df, columns=2):
    
    first_columns = df.columns
    
    if len(first_columns) < 2:
        columns = 1
        
    rows = int(np.ceil(len(first_columns) / columns) )
    figsize = (6*columns,rows*4)
    fig, ax = plt.subplots(rows, columns, figsize=figsize, squeeze=False)
    
    labels = df.index
    x = np.arange(len(labels))

    for i, col in enumerate(first_columns):

        idx = (int(i/columns), int(i%columns))

        values = df[col]
        labels =  df.index
        
        std = values.std() if values.std() > 0 else values[0] * 0.1
        max_val, min_val = values.max(), values.min()
        max_ylim = max_val + std if max_val + std < 1 else 1
        min_ylim = min_val - std if min_val - std > 0 else 0
        
        ax[idx].set_ylim(min_ylim, max_ylim)

        x = np.arange(len(values))

        temp = ax[idx].bar(x, values, 0.6, color=colors[:len(x)], alpha=0.9, tick_label=labels)

        ax[idx].set_xticks(x)
        ax[idx].set_xticklabels(labels, rotation=45)

        ax[idx].set_xlabel('method')
        ax[idx].set_ylabel('score')
        ax[idx].set_title(col)

        ax[idx].bar_label(temp, label_type='edge')

    fig.tight_layout()
    

def plot_bars_multicolumn(df, columns=2, suptitle='', switch=False, outside=True, label=False, y_lim=None, large=False):
    
    first_columns = list(dict.fromkeys([first for first, _ in df.columns]))
    second_columns = list(dict.fromkeys([second for _, second in df.columns]))
    
    if len(first_columns) < 2:
        columns = 1
    
    rows = int(np.ceil(len(first_columns) / columns) )
    figsize = (12*columns,rows*5) if large else (6*columns,rows*3)  # (6*columns,rows*4) 
    
    fig, ax = plt.subplots(rows, columns, figsize=figsize, squeeze=False)
    
    for i, col in enumerate(first_columns):
        idx = (int(i/columns), int(i%columns))

        subframe = df[col]
        
        if switch:
            subframe = subframe.T
        
        labels = subframe.index
        x = np.arange(len(labels))
        
        sublabels = subframe.columns
        
        width = 1 / len(sublabels) * 0.9

        adjusting_value = (width / 2) + ((len(sublabels)-2) / 2) * width
        x_start = x - adjusting_value
        
        if y_lim is None:
            max_val, min_val, std_val = subframe.stack().max(), subframe.stack().min(), subframe.stack().std()
            max_ylim = max_val + std_val if max_val + std_val < 1 else 1
            min_ylim = min_val - std_val if min_val - std_val > 0 else 0
        else:
            min_ylim, max_ylim = y_lim[0], y_lim[1]
        
        ax[idx].set_ylim(min_ylim, max_ylim)
        
        for i, sublabel in enumerate(sublabels):
            values = subframe[sublabel]
            
            print(sublabel)
            temp = ax[idx].bar(x_start + i*width, values, width, label=sublabel, color=colors[i], alpha=0.9) # edgecolor='black', linewidth=0.5

            if label:
                labels_text = [str(np.round(val, 3))[1:] for val in values]
                labels_text = [val + '0' if len(val) < 3 else val for val in labels_text]
                ax[idx].bar_label(temp, labels=labels_text, label_type='edge', rotation=0, padding=1.2)
                
        ax[idx].set_xticks(x)
        ax[idx].set_xticklabels([lab for lab in list(labels)])
        ax[idx].set_ylabel(col, fontsize=14)
        
        ax[idx].spines['top'].set_visible(False)
        ax[idx].spines['right'].set_visible(False)
        
        ax[idx].margins(x=0.03, y=-0)

    handles, labels = ax[idx].get_legend_handles_labels()
    
    if outside:
        bbox_to_anchor = (1.13, 0.89)  # (1.13, 0.89)
    else:
        bbox_to_anchor = (0.4, 0.9) # (0.235, 0.6)   (0.4, 0.9) (0.7, 0.4)
       
    fig.legend(handles, labels, bbox_to_anchor=bbox_to_anchor, facecolor='white', framealpha=1)  # framealpha=0.8
    
#     fig.legend(handles, labels, loc='lower center', facecolor='white', framealpha=1)  # framealpha=0.8
    
    fig.suptitle(suptitle)
    fig.tight_layout()




def plot_random_samples(dataset, num_samples=10, suptitle='test', seed=42, columns=10, clean=False):
    
    columns = columns
    rows = int(np.ceil(num_samples / columns))
    figsize = (2*columns,rows*3)
    fig, ax = plt.subplots(rows, columns, figsize=figsize, squeeze=False)
    
    inverse_norm = get_inverse_normalization(dataset, return_vals=False)
    
    rand_indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))
    
    for i, index in enumerate(rand_indices[:num_samples]):
        
        idx = (int(i/columns), int(i%columns))
                
        img = inverse_norm(dataset[index][0]).permute(1,2,0).detach().cpu().numpy()

        ax[idx].imshow(img, cmap='gray')
        ax[idx].axis('off')
        
        if not clean:
            ax[idx].set_title(str(int(index)), fontsize=20)
    
    if not clean:
        fig.suptitle(suptitle, fontsize=20)
        
    fig.tight_layout()


def plot_class_samples(dataset, labels, num_classes, samples_per_class=1, suptitle='test'):
    
    columns = 10
    rows = int(np.ceil((num_classes*samples_per_class) / columns))
    figsize = (2*columns,rows*3)
    fig, ax = plt.subplots(rows, columns, figsize=figsize, squeeze=False)
    
    inverse_norm = get_inverse_normalization(dataset, return_vals=False)
    
    for i in range(num_classes):
        
        idx = (int(i/columns), int(i%columns))
        
        mask = (torch.argmax(labels, dim=1) == i) if len(labels.shape) > 1 else (labels == i) 
        indices = torch.arange(0, len(dataset))[mask]
        
        if len(indices) > 0:
            for j in range(samples_per_class):
                
                index = indices[j] if j < len(indices) else len(indices) - 1
                
                img = inverse_norm(dataset[index][0]).permute(1,2,0).detach().cpu().numpy()
            
                ax[idx].imshow(img, cmap='gray')
                ax[idx].set_title(str(i), fontsize=20)
    
    fig.suptitle(suptitle, fontsize=20)
    fig.tight_layout()
    

def plot_samples_sorted(dataset, scores, predictions, num_samples=20, jump_size=1, descending=False, columns=10, 
                        suptitle='test', model_unc=None, decoder=None, forward_type='regular', use_layers=(0,None),
                        device='cuda:0', clean=True):
    
    rows = int(np.ceil((num_samples) / columns))
    figsize = (2*columns,rows*3)
    fig, ax = plt.subplots(rows, columns, figsize=figsize, squeeze=False)
    
    inverse_norm = get_inverse_normalization(dataset, return_vals=False)
    
    indices = range(0, len(scores), jump_size)[:num_samples]
    sorted_indices = torch.argsort(scores, dim=0, descending=descending)[indices].detach().cpu().numpy()
    sorted_scores = scores[sorted_indices]
    sorted_pred = predictions[sorted_indices]
    
    if model_unc is not None and decoder is not None:
        model_unc.to(device)
        decoder.to(device)
    
    for i, (index, score, pred) in enumerate(zip(sorted_indices, sorted_scores, sorted_pred)):

        idx = (int(i/columns), int(i%columns))
        
        img, target = dataset[index]
        
        if model_unc is not None and decoder is not None:
            
            if forward_type == 'layers':
                features = model_unc.feature_list(img.unsqueeze(0).to(device))
                features = features[use_layers[0]:use_layers[1]]
                features = [feature.squeeze(dim=1) for feature in features]
                z_latent = torch.cat(features, dim=1)
            else:
                z_latent = model_unc(img.unsqueeze(0).to(device))
                
            img = decoder(z_latent).squeeze(0).permute(1,2,0).detach().cpu().numpy()
        
        else:
            img = inverse_norm(img).permute(1,2,0).detach().cpu().numpy()
        
        score = "%.3f" % (np.round(score.detach().cpu().numpy(), 3))
        pred = pred.detach().cpu().numpy()
        
        ax[idx].imshow(img, cmap='gray')
        ax[idx].axis('off')
        ax[idx].text(0,-2, str(score) + ' / ' + str(pred), fontsize=15)
        # ax[idx].text(int(img.shape[1])-3 ,-2, str(pred), fontsize=15) # int(img.shape[1] * 0.75)
    
    if not clean:
        fig.suptitle(suptitle, fontsize=20)

    fig.tight_layout()
    
    return fig





    
