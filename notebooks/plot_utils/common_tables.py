import numpy as np
import pandas as pd
import scipy.stats as scst


def collect_data_for_final_table(model_list, res_dc, valid_pretty, horizon_list,
                                 task_ids_sorted_by_num_features, min_diff_dc,
                                 print_per_seed_results=False):
    # Generate data for final table and statistical testing
    tab_data = {}

    stat_test_data = {}
    for horizon in horizon_list:
        tab_data[horizon] = {}
        stat_test_data[horizon] = {}
        tab_data["STD %s" % horizon] = {}

        for mode in model_list[horizon]:
            assert mode in valid_pretty[horizon], (mode, valid_pretty[horizon].keys())
            # Use label, not actual key
            task_scores = []
            seed_means = []
            # Get means per tid
            for tid in task_ids_sorted_by_num_features:
                tmp = pd.DataFrame(res_dc[horizon][tid][mode]).sort_index(axis=1).ffill(axis=1).iloc[:, -1]
                assert tmp.shape == (10,), tmp.shape
                tmp = (tmp - min_diff_dc[tid][0]) / min_diff_dc[tid][1]
                task_scores.append(tmp.mean())
            # Get vars per seed
            for s in range(10):
                vals_for_this_seed = []
                for tid in task_ids_sorted_by_num_features:
                    tmp_key = sorted(list(res_dc[horizon][tid][mode][s].keys()))[-1]
                    tmp = res_dc[horizon][tid][mode][s][tmp_key]
                    tmp = (tmp - min_diff_dc[tid][0]) / min_diff_dc[tid][1]
                    vals_for_this_seed.append(tmp)
                seed_means.append(np.mean(vals_for_this_seed))
            if print_per_seed_results:
                print(valid_pretty[horizon][mode], seed_means)
            seed_means = np.array(seed_means)
            tab_data[horizon][valid_pretty[horizon][mode]] = np.round(np.mean(task_scores) * 100, 2)
            stat_test_data[horizon][valid_pretty[horizon][mode]] = task_scores
            tab_data["STD %s" % horizon][valid_pretty[horizon][mode]] = np.round(
                np.std(seed_means * 100), 2)
    return tab_data, stat_test_data


def do_wilcoxon_test(stat_test_data, model_list, horizon_list, valid_pretty, alpha=0.05, exclude=()):
    not_different = {}
    for h in horizon_list:
        not_different[h] = []
        best = 100
        best_m = None
        for m in model_list[h]:
            if exclude and (m[0] in exclude or m in exclude):
                continue
            tmp = stat_test_data[h][valid_pretty[h][m]]
            if np.mean(tmp) < best:
                best = np.mean(tmp)
                best_m = m
        for m in model_list[h]:
            if exclude and (m[0] in exclude or m in exclude):
                continue
            if m == best_m:
                continue
            chal = np.array(stat_test_data[h][valid_pretty[h][m]]) * 100
            opt = np.array(stat_test_data[h][valid_pretty[h][best_m]]) * 100
            wins = np.sum([o < c for o, c in zip(opt, chal)])
            losses = np.sum([o > c for o, c in zip(opt, chal)])
            s, p = scst.wilcoxon(x=opt, y=chal, alternative="less")

            if p > alpha:
                #print(
                #    "Not: %.3g vs %.3g; p=%.3g, s=%.3g, wins=%d, losses=%d" % (
                #    np.mean(opt), np.mean(chal), p, s, wins, losses)
                #)
                not_different[h].append((valid_pretty[h][best_m], valid_pretty[h][m], p))
            else:
                #print(
                #    "%.3g vs %.3g; p=%.3g, s=%.3g, wins=%d, losses=%d" % (
                #    np.mean(opt), np.mean(chal), p, s, wins, losses)
                #)
                pass

    for h in horizon_list:
        print("Not different with %d mins:\n\t" % h,
              "\n\t".join(["%s vs %s: %g" % n for n in not_different[h]]))
    return not_different
