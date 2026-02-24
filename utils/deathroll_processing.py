
import pandas as pd
import numpy as np
import math
import os
import ast
import random
import io
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure


def format_num(num, decimals=0):
    if num is None or (isinstance(num, float) and (math.isnan(num) or math.isinf(num))):
        return "N/A"
    if isinstance(num, (int, float)):
        if abs(num - round(num)) < 1e-9:
            return str(int(round(num)))
        return f"{num:.{decimals}f}"
    return str(num)


def calculate_global_stats(df, guild_id):
    stats = {}

    # Number of games
    stats['global_games'] = len(df)

    # --- Basic Global Roll Stats ---
    # Ensure 'rolls' is numeric first
    df['rolls'] = pd.to_numeric(df['rolls'], errors='coerce')
    # Work with rows that have valid rolls
    valid_rolls_df = df.dropna(subset=['rolls'])

    stats['max_rolls'] = None
    stats['min_rolls'] = None
    stats['max_roll_jump_url'] = "#"
    stats['min_roll_jump_url'] = "#"
    stats['average_rolls'] = 0
    stats['max_twos_count'] = 0
    stats['max_twos_jump_url'] = "#"

    if not valid_rolls_df.empty:
        idx_max_rolls = valid_rolls_df['rolls'].idxmax()
        idx_min_rolls = valid_rolls_df['rolls'].idxmin()
        row_with_max_rolls = valid_rolls_df.loc[idx_max_rolls]
        row_with_min_rolls = valid_rolls_df.loc[idx_min_rolls]
        stats['max_rolls'] = row_with_max_rolls['rolls']
        stats['min_rolls'] = row_with_min_rolls['rolls']
        stats['average_rolls'] = valid_rolls_df['rolls'].mean()

        max_roll_channel = row_with_max_rolls['channel']
        max_roll_message = row_with_max_rolls['message']
        stats['max_roll_jump_url'] = f'https://discord.com/channels/{guild_id}/{max_roll_channel}/{max_roll_message}'

        min_roll_channel = row_with_min_rolls['channel']
        min_roll_message = row_with_min_rolls['message']
        stats['min_roll_jump_url'] = f'https://discord.com/channels/{guild_id}/{min_roll_channel}/{min_roll_message}'

    # --- Calculate Global Sequence Bests ---
    global_max_prev_to_loss_num = None
    global_max_prev_to_loss_player_id = None
    global_min_ratio = float('inf')
    global_min_ratio_player_id = None
    global_min_ratio_prev_num = None
    global_min_ratio_curr_num = None
    global_max_matching_roll = None
    global_max_matching_roll_player_id = None
    global_two_after_two_counts = {}
    global_one_after_two_counts = {}

    for index, game_row in df.iterrows():  # Iterate through ALL games
        sequence_str = game_row.get('sequence')
        p1_id = game_row['player1']
        p2_id = game_row['player2']
        loser_id = game_row['loser']

        # Basic checks and parsing
        if not isinstance(sequence_str, str) or not sequence_str:
            continue
        try:
            seq_numbers = [float(n) for n in sequence_str.split('|')]
        except (ValueError, TypeError):
            continue
        num_elements = len(seq_numbers)
        if num_elements < 2:
            continue

        # --- 1. Biggest Loss (Highest pre-'1' roll for any loser) ---
        try:
            second_last_num = seq_numbers[-2]
            # Ensure the last number was indeed 1, as per rule
            if seq_numbers[-1] == 1.0 and not math.isnan(second_last_num) and not math.isinf(second_last_num):
                if global_max_prev_to_loss_num is None or second_last_num > global_max_prev_to_loss_num:
                    global_max_prev_to_loss_num = second_last_num
                    global_max_prev_to_loss_player_id = loser_id
        except IndexError:
            pass  # Sequence was too short

        # --- Determine assignment rule for pairs ---
        is_odd_length = (num_elements % 2 != 0)
        p1_gets_odd_indices = (loser_id == p2_id and is_odd_length) or \
                              (loser_id == p1_id and not is_odd_length)

        # --- 2 & 3: Loop through pairs for Min Ratio and Max Matching ---
        for i in range(1, num_elements):
            prev_num = seq_numbers[i-1]
            curr_num = seq_numbers[i]

            if math.isnan(prev_num) or math.isinf(prev_num) or \
               math.isnan(curr_num) or math.isinf(curr_num):
                continue

            # 1. Determine owner
            index_is_odd = (i % 2 != 0)
            current_owner_is_p1 = (p1_gets_odd_indices == index_is_odd)
            current_owner_id = p1_id if current_owner_is_p1 else p2_id

            # 2. Lowest % Roll
            if prev_num != 0:
                ratio = curr_num / prev_num
                if not math.isnan(ratio) and not math.isinf(ratio):
                    if ratio < global_min_ratio:
                        global_min_ratio = ratio
                        global_min_ratio_player_id = current_owner_id
                        global_min_ratio_prev_num = prev_num
                        global_min_ratio_curr_num = curr_num

            # 3. Highest 100% Roll (Matching Roll)
            if prev_num == curr_num:
                # Check if current max is None OR if curr_num is greater
                if global_max_matching_roll is None or curr_num > global_max_matching_roll:
                    global_max_matching_roll = curr_num
                    global_max_matching_roll_player_id = current_owner_id

            # 4. 2 after 2 & 1 after 2 logic
            if prev_num == 2:
                if curr_num == 2:
                    global_two_after_two_counts[current_owner_id] = global_two_after_two_counts.get(
                        current_owner_id, 0) + 1
                elif curr_num == 1:
                    global_one_after_two_counts[current_owner_id] = global_one_after_two_counts.get(
                        current_owner_id, 0) + 1

    stats['global_max_prev_to_loss_num'] = global_max_prev_to_loss_num
    stats['global_max_prev_to_loss_player_id'] = global_max_prev_to_loss_player_id
    stats['global_min_ratio'] = global_min_ratio
    stats['global_min_ratio_player_id'] = global_min_ratio_player_id
    stats['global_min_ratio_prev_num'] = global_min_ratio_prev_num
    stats['global_min_ratio_curr_num'] = global_min_ratio_curr_num
    stats['global_max_matching_roll'] = global_max_matching_roll
    stats['global_max_matching_roll_player_id'] = global_max_matching_roll_player_id
    stats['global_two_after_two_counts'] = global_two_after_two_counts
    stats['global_one_after_two_counts'] = global_one_after_two_counts

    # --- Calculate Per-Player Streaks (and global longest for special stats) ---
    # For global longest streaks, to ensure the *first* occurrence
    global_longest_win_streak = 0
    global_longest_win_streak_player_id = None
    # Initialize with a very late date
    global_longest_win_streak_datetime = pd.Timestamp.max

    global_longest_loss_streak = 0
    global_longest_loss_streak_player_id = None
    # Initialize with a very late date
    global_longest_loss_streak_datetime = pd.Timestamp.max

    # Dictionaries to store per-player streak info
    current_win_streaks = {}    # {player_id: current_streak}
    max_player_win_streaks = {}  # {player_id: max_streak_achieved}
    current_loss_streaks = {}
    max_player_loss_streaks = {}

    if not df.empty and 'winner' in df.columns and 'loser' in df.columns:
        df['winner'] = pd.to_numeric(
            df['winner'], errors='coerce').astype('Int64')
        df['loser'] = pd.to_numeric(
            df['loser'], errors='coerce').astype('Int64')
        df_sorted = df.sort_values(by='datetime').reset_index(drop=True)

        all_involved_players = pd.unique(pd.concat(
            [df_sorted['player1'], df_sorted['player2'], df_sorted['winner'], df_sorted['loser']]).dropna())
        for player_id_init in all_involved_players:
            current_win_streaks[player_id_init] = 0
            max_player_win_streaks[player_id_init] = 0
            current_loss_streaks[player_id_init] = 0
            max_player_loss_streaks[player_id_init] = 0

        for index, game_row in df_sorted.iterrows():
            winner_id = game_row['winner']
            loser_id = game_row['loser']
            # Datetime of the current game
            game_datetime = game_row['datetime']

            if pd.isna(winner_id) or pd.isna(loser_id):
                continue

            # --- Winner's Streak Updates ---
            current_loss_streaks[winner_id] = 0
            current_win_streaks[winner_id] = current_win_streaks.get(
                winner_id, 0) + 1
            max_player_win_streaks[winner_id] = max(
                max_player_win_streaks.get(winner_id, 0), current_win_streaks[winner_id])

            # Check/Update Global Longest Win Streak
            if current_win_streaks[winner_id] > global_longest_win_streak:
                global_longest_win_streak = current_win_streaks[winner_id]
                global_longest_win_streak_player_id = winner_id
                global_longest_win_streak_datetime = game_datetime
            elif current_win_streaks[winner_id] == global_longest_win_streak:
                if game_datetime < global_longest_win_streak_datetime:  # Achieved same length earlier
                    global_longest_win_streak_player_id = winner_id
                    global_longest_win_streak_datetime = game_datetime

            # --- Loser's Streak Updates ---
            current_win_streaks[loser_id] = 0
            current_loss_streaks[loser_id] = current_loss_streaks.get(
                loser_id, 0) + 1
            max_player_loss_streaks[loser_id] = max(
                max_player_loss_streaks.get(loser_id, 0), current_loss_streaks[loser_id])

            # Check/Update Global Longest Loss Streak
            if current_loss_streaks[loser_id] > global_longest_loss_streak:
                global_longest_loss_streak = current_loss_streaks[loser_id]
                global_longest_loss_streak_player_id = loser_id
                global_longest_loss_streak_datetime = game_datetime
            elif current_loss_streaks[loser_id] == global_longest_loss_streak:
                if game_datetime < global_longest_loss_streak_datetime:  # Achieved same length earlier
                    global_longest_loss_streak_player_id = loser_id
                    global_longest_loss_streak_datetime = game_datetime

    stats['global_longest_win_streak'] = global_longest_win_streak
    stats['global_longest_win_streak_player_id'] = global_longest_win_streak_player_id
    stats['global_longest_loss_streak'] = global_longest_loss_streak
    stats['global_longest_loss_streak_player_id'] = global_longest_loss_streak_player_id

    # --- Calculate Game with Most '2's ---
    if 'sequence' in df.columns and not df.empty:
        # Count occurrences of the string '2' in each sequence
        # Fill NaN sequences with empty string, convert to string just in case
        df['twos_count'] = df['sequence'].fillna('').astype(
            str).str.split('|').apply(lambda x: x.count('2'))

        if df['twos_count'].max() > 0:  # Check if any '2's were found
            # Get index of first max occurrence
            idx_max_twos = df['twos_count'].idxmax()
            stats['max_twos_count'] = df.loc[idx_max_twos, 'twos_count']
            row_with_max_twos = df.loc[idx_max_twos]

            # Construct jump URL
            try:  # Use same guild_id as obtained above
                max_twos_channel = row_with_max_twos['channel']
                max_twos_message = row_with_max_twos['message']
                # Ensure channel/message IDs are valid before creating URL
                if pd.notna(max_twos_channel) and pd.notna(max_twos_message):
                    # Cast to int
                    stats['max_twos_jump_url'] = f'https://discord.com/channels/{guild_id}/{int(max_twos_channel)}/{int(max_twos_message)}'
            except Exception as e:
                print(
                    f"Warning: Error creating jump URL for max twos: {e}")
                stats['max_twos_jump_url'] = "#"  # Reset on error

    # --- Starting Player Win % ---
    df['rolls'] = pd.to_numeric(df['rolls'], errors='coerce')
    valid_df = df.dropna(subset=['rolls', 'winner', 'loser'])
    game_count = len(valid_df)
    start_wins = 0
    for _, row in valid_df.iterrows():
        if row['rolls'] % 2 != 0:
            starting_player = row['loser']
        else:
            starting_player = row['winner']
        if row['winner'] == starting_player:
            start_wins += 1
    if game_count > 0:
        stats['start_player_win_pct'] = round(start_wins / game_count * 100, 1)
        stats['second_player_win_pct'] = round(
            (game_count - start_wins) / game_count * 100, 1)
    else:
        stats['start_player_win_pct'] = 0
        stats['second_player_win_pct'] = 0

    # --- Player Ranking Calculation ---
    all_player_ids_for_ranking = pd.concat(
        [df['player1'], df['player2']]).dropna().astype(int).unique()
    player_stats_list = []

    for pid in all_player_ids_for_ranking:
        games_as_p1 = (df['player1'] == pid).sum()
        games_as_p2 = (df['player2'] == pid).sum()
        total_p_games = games_as_p1 + games_as_p2
        if total_p_games == 0:
            continue

        total_p_wins = (df['winner'] == pid).sum()
        total_p_losses = (df['loser'] == pid).sum()

        win_pct = (total_p_wins / total_p_games) * \
            100 if total_p_games > 0 else 0.0

        p_max_w_streak = max_player_win_streaks.get(pid, 0)
        p_max_l_streak = max_player_loss_streaks.get(pid, 0)
        p_curr_w_streak = current_win_streaks.get(pid, 0)
        p_curr_l_streak = current_loss_streaks.get(pid, 0)

        current_streak_display = ""
        if p_curr_w_streak > 0:
            current_streak_display = f"w{p_curr_w_streak}"
        elif p_curr_l_streak > 0:
            current_streak_display = f"l{p_curr_l_streak}"

        streaks_combined_str = f"{current_streak_display} (W{p_max_w_streak}-L{p_max_l_streak})"

        player_stats_list.append({
            'player_id': pid,
            'total_games': total_p_games,
            'total_wins': total_p_wins,
            'total_losses': total_p_losses,
            'win_percentage': win_pct,
            'streaks_display': streaks_combined_str
        })

    stats['player_stats_list'] = player_stats_list
    return stats


def calculate_player_stats(df, player_id):
    stats = {}

    player_games = df[
        (df['player1'] == player_id) | (
            df['player2'] == player_id)
    ].copy()

    stats['total_games'] = len(player_games)
    stats['total_wins'] = (df['winner'] == player_id).sum()
    stats['total_losses'] = stats['total_games'] - stats['total_wins']

    if stats['total_games'] == 0:
        stats['win_percentage'] = "0%"
    else:
        win_percentage = (stats['total_wins'] / stats['total_games']) * 100
        stats['win_percentage'] = '{:.0f}%'.format(win_percentage)

    stats['total_rolls'] = player_games['rolls'].sum()
    avg_rolls = player_games['rolls'].mean()
    stats['average_rolls'] = '{:.2f}'.format(avg_rolls)
    stats['max_rolls'] = player_games['rolls'].max()
    stats['min_rolls'] = player_games['rolls'].min()

    # Opponents
    player_games['opponent'] = player_games.apply(
        lambda row: row['player2'] if row['player1'] == player_id else row['player1'], axis=1
    )

    stats['top_opponent_id'] = None
    stats['top_opponent_count_str'] = "0"
    stats['top_victim_id'] = None
    stats['top_victim_difference'] = 0
    stats['top_nemesis_id'] = None
    stats['top_nemesis_difference'] = 0

    if not player_games.empty and 'opponent' in player_games.columns:
        opponent_counts = player_games['opponent'].value_counts()
        if not opponent_counts.empty:
            stats['top_opponent_id'] = opponent_counts.idxmax()
            stats['top_opponent_count_str'] = str(opponent_counts.max())

        wins_vs_opponent = player_games[player_games['winner']
                                        == player_id]['opponent'].value_counts()
        losses_vs_opponent = player_games[player_games['loser']
                                          == player_id]['opponent'].value_counts()

        matchup_stats_df = pd.DataFrame({
            'wins_vs': wins_vs_opponent,
            'losses_vs': losses_vs_opponent
        }).fillna(0).astype(int)

        if not matchup_stats_df.empty:
            matchup_stats_df['difference'] = matchup_stats_df['wins_vs'] - \
                matchup_stats_df['losses_vs']

            positive_diffs = matchup_stats_df[matchup_stats_df['difference'] > 0]
            if not positive_diffs.empty:
                stats['top_victim_id'] = positive_diffs['difference'].idxmax()
                stats['top_victim_difference'] = positive_diffs.loc[stats['top_victim_id']]['difference']

            negative_diffs = matchup_stats_df[matchup_stats_df['difference'] < 0]
            if not negative_diffs.empty:
                stats['top_nemesis_id'] = negative_diffs['difference'].idxmin()
                stats['top_nemesis_difference'] = negative_diffs.loc[stats['top_nemesis_id']]['difference']

    # Roll ratios
    biggest_loss_numbers = []
    matching_rolls_list = []
    player_roll_ratios_all_games = []
    min_ratio_so_far = float('inf')
    player_two_after_two_count = 0
    player_one_after_two_count = 0
    min_prev_num_for_ratio = 0
    min_curr_num_for_ratio = 0

    for index, game_row in player_games.iterrows():
        sequence_str = game_row.get('sequence')
        p1_id = game_row['player1']
        p2_id = game_row['player2']
        loser_id = game_row['loser']
        is_player_loser = (loser_id == player_id)

        if not isinstance(sequence_str, str) or not sequence_str:
            continue

        seq_numbers = [float(n) for n in sequence_str.split('|')]
        num_elements = len(seq_numbers)

        is_odd_length = (num_elements % 2 != 0)
        p1_gets_odd_indices = (loser_id == p2_id and is_odd_length) or \
            (loser_id == p1_id and not is_odd_length)

        for i in range(1, num_elements):
            prev_num = seq_numbers[i-1]
            curr_num = seq_numbers[i]
            index_is_odd = (i % 2 != 0)
            current_owner_is_p1 = (p1_gets_odd_indices == index_is_odd)
            current_owner_id = p1_id if current_owner_is_p1 else p2_id

            if current_owner_id == player_id and prev_num != 0:
                ratio = curr_num / prev_num
                player_roll_ratios_all_games.append(ratio)
                if ratio < min_ratio_so_far:
                    min_ratio_so_far = ratio
                    min_prev_num_for_ratio = prev_num
                    min_curr_num_for_ratio = curr_num
                if prev_num == 2:
                    if curr_num == 2:
                        player_two_after_two_count += 1
                    elif curr_num == 1:
                        player_one_after_two_count += 1

            if prev_num == curr_num and current_owner_id == player_id:
                matching_rolls_list.append(curr_num)

        if is_player_loser:
            try:
                second_last_str = sequence_str.split('|')[-2]
                second_last_num = int(second_last_str)
                biggest_loss_numbers.append(second_last_num)
            except (ValueError, IndexError):
                pass

    stats['biggest_loss'] = max(biggest_loss_numbers) if biggest_loss_numbers else "N/A"
    stats['min_ratio'] = '{:.2f}%'.format(min_ratio_so_far * 100) if min_ratio_so_far != float('inf') else "N/A"
    stats['min_prev_num_for_ratio'] = '{:.0f}'.format(min_prev_num_for_ratio)
    stats['min_curr_num_for_ratio'] = '{:.0f}'.format(min_curr_num_for_ratio)
    stats['max_match'] = '{:.0f}'.format(max(matching_rolls_list)) if matching_rolls_list else "N/A"

    stats['avg_player_roll_pct_str'] = "N/A"
    if player_roll_ratios_all_games:
        avg_ratio_val = sum(player_roll_ratios_all_games) / len(player_roll_ratios_all_games)
        stats['avg_player_roll_pct_str'] = '{:.2f}%'.format(avg_ratio_val * 100)

    stats['player_two_after_two_count'] = player_two_after_two_count
    stats['player_one_after_two_count'] = player_one_after_two_count

    return stats


def generate_charts(df, player_names_map):

    # COLOR configs for plots:
    figbg_c = '#121214'
    bar_c = '#0969a1'
    label_c = '#82838b'
    vs_colors = [(0, '#9f3c3c'), (0.25, '#9f3c3c'),
                 (0.5, '#9f9f3c'), (0.75, '#3c9f3c'), (1, '#3c9f3c')]

    # Create figure 1
    fig1 = Figure(figsize=(8, 6), facecolor=figbg_c)
    ax1 = fig1.subplots()
    fig1.subplots_adjust(left=0.1, right=0.88)

    # Data for Barplot
    df_grouped = df.groupby(["rolls"]).agg(count=("rolls", "count")).reset_index()

    max_rolls = df_grouped['rolls'].max()
    df_barplot = df_grouped
    for i in range(1, max_rolls+2):
        if not (df_grouped['rolls'] == i).any():
            df_append = pd.DataFrame([[i, 0]], columns=['rolls', 'count'])
            df_barplot = pd.concat([df_barplot, df_append])
    df_barplot = df_barplot.sort_values('rolls').reset_index(drop=True)

    # Data of 100k simulated games
    sim_data = {2: 2, 3: 3, 4: 31, 5: 120, 6: 288, 7: 734, 8: 1371, 9: 2618, 10: 3912,
                11: 5585, 12: 7318, 13: 8868, 14: 9689, 15: 9918, 16: 9646, 17: 8968,
                18: 7836, 19: 6311, 20: 4967, 21: 3787, 22: 2751, 23: 1826, 24: 1286,
                25: 861, 26: 548, 27: 334, 28: 194, 29: 96, 30: 59, 31: 33, 32: 19,
                33: 12, 34: 6, 35: 2, 37: 1}
    simulation = pd.Series(data=sim_data)
    sim_total_rolls = simulation.sum()
    sim_roll_percentages = (simulation / sim_total_rolls) * 100

    # Barplot
    ax1.bar(df_barplot['rolls'], df_barplot['count'], color=bar_c, edgecolor='none', label='Actual Games')
    ax1.set_xlabel('rolls', color=label_c)
    ax1.set_ylabel('count', color=label_c)
    ax1.set_title('How many rolls?', color=label_c)
    ax1.tick_params(axis='x', colors=label_c)
    ax1.tick_params(axis='y', colors=label_c)
    ax1.set_xticks(range(1, max_rolls + 1))
    ax1.yaxis.set_major_locator(mticker.MaxNLocator(nbins=20, integer=True))
    ax1.set_facecolor(figbg_c)
    ax1.grid(color=label_c, linestyle='-', linewidth=0.5, alpha=0.5)
    ax1.spines[:].set_color(label_c)
    ax1.set_xlim(xmin=0, xmax=max_rolls+1)

    # Normalize simulation to total actual games
    total_real_games = df_barplot['count'].sum()
    sim_normalized_counts = (sim_roll_percentages / 100) * total_real_games

    # Plot the simulated curve on the same axis
    ax1.plot(sim_normalized_counts.index, sim_normalized_counts.values,
             marker='', linestyle='-', color='#9f3c3c', label='simulated distribution')

    # Optionally add a legend if helpful, or just rely on the existing colors
    ax1.legend(labelcolor=label_c, facecolor=figbg_c,
               edgecolor=label_c, fontsize='small')

    # Cumulative Win/Loss Line Plot per Player
    df_sorted = df.sort_values(by='datetime').reset_index(drop=True)
    all_players = sorted(list(set(df['player1']) | set(df['player2'])))

    # Build cumulative score per player: +1 for win, -1 for loss
    player_cumulative = {}
    for pid in all_players:
        player_games = df_sorted[
            (df_sorted['player1'] == pid) | (df_sorted['player2'] == pid)
        ]
        cumulative = [0]
        for _, game_row in player_games.iterrows():
            if game_row['winner'] == pid:
                cumulative.append(cumulative[-1] + 1)
            else:
                cumulative.append(cumulative[-1] - 1)
        player_cumulative[pid] = cumulative

    max_games = max(len(c) - 1 for c in player_cumulative.values()
                    ) if player_cumulative else 0

    # Distinct color palette for player lines
    line_colors = [
        '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
        '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990',
        '#dcbeff', '#9A6324', '#fffac8', '#800000', '#aaffc3',
        '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#ffffff'
    ]

    fig2 = Figure(figsize=(8, 6), facecolor=figbg_c)
    ax2 = fig2.subplots()
    fig2.subplots_adjust(left=0.1, right=0.95, bottom=0.15)

    for i, pid in enumerate(all_players):
        cumulative = player_cumulative[pid]
        color = line_colors[i % len(line_colors)]
        label = player_names_map.get(pid, f"ID:{pid}")
        ax2.plot(range(len(cumulative)), cumulative, marker='', linestyle='-',
                 color=color, label=label, linewidth=1.5, alpha=0.9)

    ax2.set_title('Cumulative Wins/Losses per Player', color=label_c)
    ax2.set_xlabel('games played', color=label_c)
    ax2.set_ylabel('cumulative score (win +1 / loss -1)', color=label_c)
    ax2.set_facecolor(figbg_c)
    ax2.tick_params(axis='x', colors=label_c)
    ax2.tick_params(axis='y', colors=label_c)
    ax2.set_xlim(0, max_games)
    ax2.xaxis.set_major_locator(mticker.MaxNLocator(
        integer=True, steps=[1, 2, 5, 10]))
    ax2.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax2.axhline(y=0, color=label_c, linestyle='-', linewidth=0.5, alpha=0.3)
    ax2.grid(color=label_c, linestyle='-', linewidth=0.5, alpha=0.3)
    ax2.spines[:].set_color(label_c)
    ax2.legend(labelcolor=label_c, facecolor=figbg_c, edgecolor=label_c,
               fontsize='small', loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=8)

    # Duels HeatMap
    df_duels = df
    players = sorted(list(set(df['player1']) | set(df['player2'])))
    results = []
    for player1 in players:
        for player2 in players:
            if player1 != player2:
                df_filtered = df_duels[((df_duels['player1'] == player1) & (df_duels['player2'] == player2)) |
                                       ((df_duels['player1'] == player2) & (df_duels['player2'] == player1))]
                wins = len(df_filtered[df_filtered['winner'] == player1])
                losses = len(df_filtered[df_filtered['loser'] == player1])
                if (wins + losses) > 0:
                    win_percentage = wins / (wins + losses)
                    results.append({'Spieler1': player1, 'Spieler2': player2,
                                   'Siege': wins, 'Niederlagen': losses, 'Siegquote': win_percentage})
                else:
                    results.append({'Spieler1': player1, 'Spieler2': player2,
                                   'Siege': 0, 'Niederlagen': 0, 'Siegquote': np.nan})
    win_loss_df = pd.DataFrame(results)
    win_loss_pivot = win_loss_df.pivot(index='Spieler1', columns='Spieler2', values='Siegquote')

    fig3 = Figure(figsize=(8, 6), facecolor=figbg_c)
    ax3 = fig3.subplots()
    cmap_name = 'vs_colormap'
    cm = LinearSegmentedColormap.from_list(cmap_name, vs_colors, N=100)
    ax3.imshow(win_loss_pivot, cmap=cm, interpolation='nearest', vmin=0, vmax=1)

    player_labels = [player_names_map.get(pid, f"ID:{pid}") for pid in players]

    ax3.set_xticks(np.arange(len(players)))
    ax3.set_xticklabels(player_labels, color=label_c)
    ax3.set_yticks(np.arange(len(players)))
    ax3.set_yticklabels(player_labels, color=label_c)

    for i, player1 in enumerate(players):
        for j, player2 in enumerate(players):
            if i == j:
                ax3.text(j, i, 'X', ha='center', va='center', color=label_c)
                ax3.add_patch(mpatches.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=figbg_c, edgecolor='none'))
            elif np.isnan(win_loss_pivot.iloc[i, j]):
                ax3.text(j, i, '-', ha='center', va='center', color=label_c)
                ax3.add_patch(mpatches.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor='#1a1a1e', edgecolor='none'))
            else:
                wins = win_loss_df.loc[(win_loss_df['Spieler1'] == player1) & (
                    win_loss_df['Spieler2'] == player2), 'Siege'].values[0]
                losses = win_loss_df.loc[(win_loss_df['Spieler1'] == player1) & (
                    win_loss_df['Spieler2'] == player2), 'Niederlagen'].values[0]
                ax3.text(j, i, f"{win_loss_pivot.iloc[i, j]:.0%}\n({wins} - {losses})", ha='center', va='center', color='black')

    ax3.spines[:].set_visible(False)
    ax3.set_facecolor(figbg_c)
    ax3.set_xticks(np.arange(win_loss_pivot.shape[1]+1)-.5, minor=True)
    ax3.set_yticks(np.arange(win_loss_pivot.shape[0]+1)-.5, minor=True)
    ax3.xaxis.set_tick_params(which='major', color=label_c)
    ax3.yaxis.set_tick_params(which='major', color=label_c)
    ax3.grid(which="minor", color=figbg_c, linestyle='-', linewidth=3)
    ax3.tick_params(which="minor", top=False, left=False)
    ax3.set_title('Duels', color=label_c)
    ax3.xaxis.tick_top()

    # Data for Lineplot
    def get_rolls_list(sequence_string):
        return [int(roll) for roll in sequence_string.split('|')]

    df_line = df.copy()
    df_line['rolls_list'] = df_line['sequence'].apply(get_rolls_list)

    short_rolls = {}
    for rolls in df_line[df_line['rolls'] == df_line['rolls'].min()]['rolls_list']:
        for i, roll_value in enumerate(rolls):
            roll_number = i
            if roll_number not in short_rolls:
                short_rolls[roll_number] = []
            short_rolls[roll_number].append(roll_value)

    short_max_rolls = {roll: np.max(values) for roll, values in short_rolls.items()}
    short_min_rolls = {roll: np.min(values) for roll, values in short_rolls.items()}
    short_max_rolls = list(short_max_rolls.values())
    short_min_rolls = list(short_min_rolls.values())[0:-1]
    short_min_rolls.reverse()

    short_path_rolls = short_max_rolls
    short_path_rolls.extend(short_min_rolls)

    long_rolls = {}
    for rolls in df_line[df_line['rolls'] == df_line['rolls'].max()]['rolls_list']:
        for i, roll_value in enumerate(rolls):
            roll_number = i
            if roll_number not in long_rolls:
                long_rolls[roll_number] = []
            long_rolls[roll_number].append(roll_value)

    long_max_rolls = {roll: np.max(values) for roll, values in long_rolls.items()}
    long_min_rolls = {roll: np.min(values) for roll, values in long_rolls.items()}
    long_max_rolls = list(long_max_rolls.values())
    long_min_rolls = list(long_min_rolls.values())[0:-1]
    long_min_rolls.reverse()

    long_path_rolls = long_max_rolls
    long_path_rolls.extend(long_min_rolls)

    min_patch, max_patch = None, None
    min1, max1 = False, False
    shortest_game_rolls, longest_game_rolls = [], []

    if len(df_line[df_line['rolls'] == df_line['rolls'].min()]) == 1:
        min1 = True
        shortest_game_index = df_line['rolls'].idxmin()
        shortest_game_rolls = df_line['rolls_list'][shortest_game_index]
    elif len(df_line[df_line['rolls'] == df_line['rolls'].min()]) > 1:
        min1 = False
        minPathData = []
        for x, y in enumerate(short_path_rolls):
            if x == 0:
                minPathPoint = (mpath.Path.MOVETO, [x, y])
            elif x == len(short_path_rolls)-1:
                minPathPoint = (mpath.Path.CLOSEPOLY, [0, 0])
            else:
                if x > (len(short_path_rolls)-1)/2:
                    x = (len(short_path_rolls)-1) - x
                minPathPoint = (mpath.Path.LINETO, [x, y])
            minPathData.append(minPathPoint)
        min_codes, min_verts = zip(*minPathData)
        minPath = mpath.Path(min_verts, min_codes)
        min_patch = mpatches.PathPatch(minPath, facecolor='#9f3c3c', edgecolor=None, alpha=0.3,
                                       label=f"corridor of shortest games (Length: {df_line['rolls'].min()})")

    if len(df_line[df_line['rolls'] == df_line['rolls'].max()]) == 1:
        max1 = True
        longest_game_index = df_line['rolls'].idxmax()
        longest_game_rolls = df_line['rolls_list'][longest_game_index]
    elif len(df_line[df_line['rolls'] == df_line['rolls'].max()]) > 1:
        max1 = False
        maxPathData = []

        for x, y in enumerate(long_path_rolls):
            if x == 0:
                maxPathPoint = (mpath.Path.MOVETO, [x, y])
            elif x == len(long_path_rolls)-1:
                maxPathPoint = (mpath.Path.CLOSEPOLY, [0, 0])
            else:
                if x > (len(long_path_rolls)-1)/2:
                    x = (len(long_path_rolls)-1) - x
                maxPathPoint = (mpath.Path.LINETO, [x, y])
            maxPathData.append(maxPathPoint)
        max_codes, max_verts = zip(*maxPathData)
        maxPath = mpath.Path(max_verts, max_codes)
        max_patch = mpatches.PathPatch(maxPath, facecolor='#3c9f3c', edgecolor=None,
                                       alpha=0.3, label=f"corridor of longest games (Length: {df_line['rolls'].max()})")

    all_rolls_per_roll_number = {}

    for rolls in df_line['rolls_list']:
        for i, roll_value in enumerate(rolls):
            roll_number = i
            if roll_number not in all_rolls_per_roll_number:
                all_rolls_per_roll_number[roll_number] = []
            all_rolls_per_roll_number[roll_number].append(roll_value)

    average_rolls = {roll: np.mean(values) for roll, values in all_rolls_per_roll_number.items()}
    max_rolls = {roll: np.max(values) for roll, values in all_rolls_per_roll_number.items()}
    min_rolls = {roll: np.min(values) for roll, values in all_rolls_per_roll_number.items()}

    sorted_averages_rolls = dict(sorted(average_rolls.items()))
    sorted_max_rolls = dict(sorted(max_rolls.items()))
    sorted_min_rolls = dict(sorted(min_rolls.items()))

    fig4 = Figure(figsize=(8, 6), facecolor=figbg_c)
    ax4 = fig4.subplots()

    ax4.plot(sorted_min_rolls.keys(), sorted_min_rolls.values(), linestyle='--', label='Minimum Roll Value', alpha=0.8, color='#9f3c3c')
    if min1:
        ax4.plot(range(0, len(shortest_game_rolls)), shortest_game_rolls, marker=11, linestyle='-',
                 label=f'Shortest Game (Length: {len(shortest_game_rolls)-1})', linewidth=1, color='#9f3c3c')
    else:
        ax4.add_patch(min_patch)
    ax4.plot(sorted_max_rolls.keys(), sorted_max_rolls.values(), linestyle='--', label='Maximum Roll Value', alpha=0.8, color='#3c9f3c')
    if max1:
        ax4.plot(range(0, len(longest_game_rolls)), longest_game_rolls, marker=10, linestyle='-',
                 label=f'Longest Game (Length: {len(longest_game_rolls)-1})', linewidth=1, color='#3c9f3c')
    else:
        ax4.add_patch(max_patch)
    ax4.plot(sorted_averages_rolls.keys(), sorted_averages_rolls.values(), marker='o',
             linestyle='-', label='Average Roll Value', alpha=1, color=bar_c, markersize=3)

    ax4.set_title('Average, Max, Min, Shortest & Longest Game Roll Values', color=label_c)
    ax4.set_xlabel('roll number', color=label_c)
    ax4.set_ylabel('roll value', color=label_c)
    ax4.set_facecolor(figbg_c)
    ax4.tick_params(axis='x', colors=label_c)
    ax4.tick_params(axis='y', colors=label_c)
    ax4.set_yscale('log')
    ax4.grid(True, which="both", ls="-", color=label_c, linewidth=0.5, alpha=0.5)
    ax4.spines[:].set_color(label_c)
    ax4.set_xticks(np.arange(0, max(sorted_averages_rolls.keys()) + 1, 1))
    ax4.set_ylim(1, 200000)
    ax4.legend(labelcolor=label_c, facecolor=figbg_c, edgecolor=label_c, reverse=True)

    image_buffers = []
    for fig in [fig3, fig1, fig4, fig2]:
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=400)
        buffer.seek(0)
        image_buffers.append(buffer)

    return image_buffers, players
