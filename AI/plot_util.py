import matplotlib.pyplot as plt
# Define the groups for the first 9 keys
group1_keys = [
    ('background_up', 'up_change_pred_pct', ),
    ('background_down', 'down_change_pred_pct', ),
    ('background_none', 'none_change_pred_pct', )
]

group2_keys = [
    'up_change_pred_precision',
    'down_change_pred_precision',
    'none_change_pred_precision'
]

# Define the groups for the next 8 keys
group3_keys = [
    'accuracy',
    'change_accuracy',
    'change_precision',
    'direction_precision'
]

# Define line styles and markers for the first 9 keys
line_styles1 = ['-', '--', '-.']
markers1 = ['^', 'v', '>']
colors = ['r', 'g', 'b']

# Define line styles and markers for the next 8 keys
line_styles2 = ['-', '-', '--', '--']
markers2 = ['$A$', '$A$', '$P$', '$P$']

def plot_history(plot_hist_dict, ax_0, ax_1, ax_2):
    col_num = len(plot_hist_dict['background_up'])-1
    
    ax_0.clear()
    for i, group in enumerate(group1_keys):
        for j, key in enumerate(group):
            ax_0.plot(plot_hist_dict[key], linestyle=line_styles1[j], marker=markers1[i], label=key, color=colors[i])
            
            start_value = plot_hist_dict[key][0]
            end_value = plot_hist_dict[key][-1]
            ax_0.text(0, start_value, f'{start_value:6.2f}', fontsize=9)
            ax_0.text(col_num, end_value, f'{end_value:6.2f}', fontsize=9)
    ax_0.legend()
    ax_0.set_xlabel('Epochs')
    ax_0.set_ylabel('%')
    # ax_0.set_title('Prediction vs Background pct')



    ax_1.clear()
    for i, group in enumerate(group2_keys):
        ax_1.plot(plot_hist_dict[group], linestyle=line_styles1[0], marker=markers1[i], label=group)

        start_value = plot_hist_dict[group][0]
        end_value = plot_hist_dict[group][-1]
        ax_1.text(0, start_value, f'{start_value:6.2f}', fontsize=9)
        ax_1.text(col_num, end_value, f'{end_value:6.2f}', fontsize=9)

    ax_1.legend()
    ax_1.set_xlabel('Epochs')
    ax_1.set_ylabel('%')
    # ax_1.set_title('Prediction precision')



    ax_2.clear()
    for i, group in enumerate(group3_keys):
        ax_2.plot(plot_hist_dict[group], linestyle=line_styles2[0], marker=markers2[i], label=group)

        start_value = plot_hist_dict[group][0]
        end_value = plot_hist_dict[group][-1]
        ax_2.text(0, start_value, f'{start_value:6.2f}', fontsize=9)
        ax_2.text(col_num, end_value, f'{end_value:6.2f}', fontsize=9)

    ax_2.legend()
    ax_2.set_xlabel('Epochs')
    ax_2.set_ylabel('%')
    # ax_2.set_title('Need Change?')

    plt.pause(1)