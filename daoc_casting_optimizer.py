import pandas as pd
from pydantic import BaseModel, Field
from typing import List
from math import floor
import gradio as gr
import plotly.express as px
import plotly.graph_objects as go

class DelveData(BaseModel):
    costlevel: List[int] = Field([0, 1, 1, 2, 3, 3, 5, 5, 7, 7])
    costtotal: List[int] = Field([0, 1, 2, 4, 7, 10, 15, 20, 27, 34])
    delvemom: List[int] = Field([0, 1, 2, 3, 5, 7, 9, 11, 13, 15])
    delvewp: List[int] = Field([0, 3, 6, 9, 13, 17, 22, 27, 33, 39])
    delveaa: List[int] = Field([0, 4, 8, 12, 17, 22, 28, 34, 41, 48])

class TableUpdater:
    def __init__(self, base: int, crit: int, critmin: int, critmax: int, data: DelveData, available_points: int):
        self.base = base
        self.crit = crit
        self.critmin = critmin
        self.critmax = critmax
        self.data = data
        self.available_points = available_points
        self.levels = {"mom": 0, "wp": 0, "aa": 0}

    def update_table(self) -> pd.DataFrame:
        rows = []
        for i in range(1, 10):
            row = {
                "Level": i,
                "Cost Level": self.data.costlevel[i],
                "Cost Total": self.data.costtotal[i],
                "MOM1": self.data.delvemom[i],
                "MOM Calc": self.calculate_mom(i),  
                "MOM Diff": self.calculate_mom_difference(i),
                "MOM Ratio": self.calculate_mom_ratio(i),
                "WP1": self.data.delvewp[i],
                "WP Calc": self.calculate_wp(i),
                "WP Diff": self.calculate_wp_difference(i),
                "WP Ratio": self.calculate_wp_ratio(i),
                "AA1": self.data.delveaa[i],
                "AA Calc": self.calculate_aa(i),
                "AA Diff": self.calculate_aa_difference(i),
                "AA Ratio": self.calculate_aa_ratio(i)
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def calculate_wp(self, i: int) -> float:
        return 100 * (100 + min(50, self.data.delvewp[i] + self.crit) * (self.critmin + self.critmax) / 200) / \
               (100 + min(50, self.crit) * (self.critmin + self.critmax) / 200) - 100

    def calculate_wp_difference(self, i: int) -> float:
        wp1 = self.calculate_wp(i)
        wp2 = self.calculate_wp(i - 1)
        return (wp1 - wp2) / self.data.costlevel[i]

    def calculate_wp_ratio(self, i: int) -> float:
        return self.calculate_wp(i) / self.data.costtotal[i]

    def calculate_aa(self, i: int) -> float:
        return 100 * (100 + floor((self.base + self.data.delveaa[i]) / 2)) / (100 + floor(self.base / 2)) - 100

    def calculate_aa_difference(self, i: int) -> float:
        aa1 = self.calculate_aa(i)
        aa2 = self.calculate_aa(i - 1)
        return (aa1 - aa2) / self.data.costlevel[i]

    def calculate_aa_ratio(self, i: int) -> float:
        return self.calculate_aa(i) / self.data.costtotal[i]

    def calculate_mom(self, i: int) -> float:
        return self.data.delvemom[i]

    def calculate_mom_difference(self, i: int) -> float:
        mom1 = self.calculate_mom(i)
        mom2 = self.calculate_mom(i - 1)
        return (mom1 - mom2) / self.data.costlevel[i]

    def calculate_mom_ratio(self, i: int) -> float:
        return self.calculate_mom(i) / self.data.costtotal[i]

    def get_total_cost_and_dps(self):
        total_cost = sum([self.data.costtotal[self.levels[skill]] for skill in self.levels])
        total_dps_mom = self.calculate_mom(self.levels['mom'])
        total_dps_wp = self.calculate_wp(self.levels['wp'])
        total_dps_aa = self.calculate_aa(self.levels['aa'])
        total_dps = total_dps_mom + total_dps_wp + total_dps_aa
        return total_cost, total_dps_mom, total_dps_wp, total_dps_aa, total_dps

    def get_total_invested_points(self):
        return sum([self.data.costtotal[self.levels[skill]] for skill in self.levels])

    def get_summary(self):
        summary = []
        total_dps = 0
        total_cost = 0

        for skill in ["mom", "wp", "aa"]:
            level = self.levels[skill]
            next_level = level + 1 if level < 9 else level
            next_point_cost = self.data.costlevel[next_level] if level < 9 else 0

            if skill == "mom":
                dps = self.calculate_mom(level)
                next_dps = self.calculate_mom(next_level)
            elif skill == "wp":
                dps = self.calculate_wp(level)
                next_dps = self.calculate_wp(next_level)
            elif skill == "aa":
                dps = self.calculate_aa(level)
                next_dps = self.calculate_aa(next_level)

            next_dps_gain_per_point = (next_dps - dps) / next_point_cost if next_point_cost > 0 else 0

            summary.append({
                "Skill": skill.upper(),
                "Level": level,
                "DPS": dps,
                "Total Cost": self.data.costtotal[level],
                "Next Point Cost": next_point_cost,
                "Next DPS Gain per point": next_dps_gain_per_point
            })

            total_dps += dps
            total_cost += self.data.costtotal[level]

        summary.append({
            "Skill": "TOTAL",
            "Level": "",
            "DPS": total_dps,
            "Total Cost": total_cost,
            "Next Point Cost": "",
            "Next DPS Gain per point": ""
        })

        return pd.DataFrame(summary)

    def set_level(self, skill: str, level: int):
        level = int(level)
        if 0 <= level <= 9:
            self.levels[skill] = level
        total_cost, total_dps_mom, total_dps_wp, total_dps_aa, total_dps = self.get_total_cost_and_dps()
        df = self.update_table()
        summary_df = self.get_summary()
        return summary_df, df

    def increment_level(self, skill: str):
        total_invested_points = self.get_total_invested_points()
        if total_invested_points + self.data.costlevel[self.levels[skill] + 1] <= self.available_points:
            return self.set_level(skill, self.levels[skill] + 1)
        return self.get_summary(), self.update_table()

    def decrement_level(self, skill: str):
        return self.set_level(skill, self.levels[skill] - 1)
    
    def set_available_points(self, available_points: int):
        self.available_points = available_points

    def optimize_levels(self):
        dp = [0] * (self.available_points + 1)

        points = self.available_points
        best_combination = (0, 0, 0)
        best_dps = 0
        for mom_level in range(10):
            for wp_level in range(10):
                for aa_level in range(10):
                    total_dps = self.calculate_aa(aa_level) + self.calculate_mom(mom_level) + self.calculate_wp(wp_level)
                    total_cost = self.data.costtotal[mom_level] + self.data.costtotal[wp_level] + self.data.costtotal[aa_level]
                    if total_cost > points:
                        continue
                    else:
                        if total_dps > best_dps:
                            best_dps = total_dps
                            best_combination = (mom_level, wp_level, aa_level)
        best_points = self.available_points
        while best_points > 0 and dp[best_points] == 0:
            best_points -= 1

        best_mom, best_wp, best_aa = best_combination
        self.set_level('mom', best_mom)
        self.set_level('wp', best_wp)
        self.set_level('aa', best_aa)

        return self.get_summary(), self.update_table(), best_mom, best_wp, best_aa, self.get_total_invested_points()

    def calculate_dps_curve(self):
        dps_curve = []
        original_available_points = self.available_points

        for points in range(0, original_available_points + 1):
            self.set_available_points(points)
            self.optimize_levels()
            total_cost, total_dps_mom, total_dps_wp, total_dps_aa, total_dps = self.get_total_cost_and_dps()
            dps_curve.append((points, total_dps))

        self.set_available_points(original_available_points)  # Restore original points

        return pd.DataFrame(dps_curve, columns=["Total Investment Points", "Total DPS"])

delve_data = DelveData()
available_points = 50
table_updater = TableUpdater(base=224, crit=0, critmin=10, critmax=50, data=delve_data, available_points=available_points)

def set_available_points(available_points: int):
    table_updater.set_available_points(available_points)

def update_mom(value=None):
    if value is not None:
        summary_df, df = table_updater.set_level('mom', value)
    else:
        summary_df, df = table_updater.increment_level('mom')
    return summary_df, df, table_updater.levels['mom'], table_updater.get_total_invested_points()

def update_wp(value=None):
    if value is not None:
        summary_df, df = table_updater.set_level('wp', value)
    else:
        summary_df, df = table_updater.increment_level('wp')
    return summary_df, df, table_updater.levels['wp'], table_updater.get_total_invested_points()

def update_aa(value=None):
    if value is not None:
        summary_df, df = table_updater.set_level('aa', value)
    else:
        summary_df, df = table_updater.increment_level('aa')
    return summary_df, df, table_updater.levels['aa'], table_updater.get_total_invested_points()

def decrement_mom():
    summary_df, df = table_updater.decrement_level('mom')
    return summary_df, df, table_updater.levels['mom'], table_updater.get_total_invested_points()

def decrement_wp():
    summary_df, df = table_updater.decrement_level('wp')
    return summary_df, df, table_updater.levels['wp'], table_updater.get_total_invested_points()

def decrement_aa():
    summary_df, df = table_updater.decrement_level('aa')
    return summary_df, df, table_updater.levels['aa'], table_updater.get_total_invested_points()

def initialize_table(base, crit, critmin, critmax, available_points):
    global table_updater
    table_updater = TableUpdater(base=base, crit=crit, critmin=critmin, critmax=critmax, data=delve_data, available_points=available_points)
    summary_df, df = table_updater.get_total_cost_and_dps(), table_updater.update_table()
    summary = table_updater.get_summary()
    return summary, df, 0, 0, 0, table_updater.get_total_invested_points(), available_points

def automatic_optimization(points_available: int):
    summary_df, df, mom_level, wp_level, aa_level, invested_points = table_updater.optimize_levels()
    return summary_df, df, mom_level, wp_level, aa_level, invested_points, points_available

def create_graphs(summary_df):
    total_cost, total_dps_mom, total_dps_wp, total_dps_aa, total_dps = table_updater.get_total_cost_and_dps()
    
    # Total Investment vs. DPS
    fig_investment_dps = px.line(summary_df, x='Total Cost', y='DPS', title='Total Investment vs. DPS')
    
    # Skill Levels vs. DPS Contribution
    skills = ['MOM', 'WP', 'AA']
    levels = [table_updater.levels[skill.lower()] for skill in skills]
    dps_values = [total_dps_mom, total_dps_wp, total_dps_aa]
    fig_skills_dps = go.Figure(data=[
        go.Bar(name='DPS Contribution', x=skills, y=dps_values),
        go.Scatter(name='Skill Levels', x=skills, y=levels, mode='lines+markers', yaxis='y2')
    ])
    fig_skills_dps.update_layout(
        title='Skill Levels vs. DPS Contribution',
        yaxis=dict(title='DPS'),
        yaxis2=dict(title='Skill Levels', overlaying='y', side='right')
    )
    
    # Crit Chance vs. DPS
    crit_values = list(range(0, 101, 5))
    dps_values = [100 * (100 + min(50, table_updater.crit + crit) * (table_updater.critmin + table_updater.critmax) / 200) /
                  (100 + min(50, table_updater.crit) * (table_updater.critmin + table_updater.critmax) / 200) - 100
                  for crit in crit_values]
    fig_crit_dps = px.line(x=crit_values, y=dps_values, title='Crit Chance vs. DPS', labels={'x': 'Crit Chance', 'y': 'DPS'})
    
    return fig_investment_dps, fig_skills_dps, fig_crit_dps

def plot_dps_curve():
    dps_curve_df = table_updater.calculate_dps_curve()
    fig_dps_curve = px.line(dps_curve_df, x='Total Investment Points', y='Total DPS', title='Total DPS vs. Investment Points', height=700)
    return fig_dps_curve

with gr.Blocks() as demo:
    gr.Markdown("## Investment Table")
    
    with gr.Row():
        base_input = gr.Number(label="Base", value=224)
        crit_input = gr.Number(label="Crit", value=0)
        critmin_input = gr.Number(label="Critmin", value=10)
        critmax_input = gr.Number(label="Critmax", value=50)
    
    initialize_button = gr.Button("Initialize Table")
    
    with gr.Row():
        mom_decrement_button = gr.Button("- MOM")
        mom_input = gr.Number(label="MOM Level", value=0)
        mom_increment_button = gr.Button("+ MOM")
    
    with gr.Row():
        wp_decrement_button = gr.Button("- WP")
        wp_input = gr.Number(label="WP Level", value=0)
        wp_increment_button = gr.Button("+ WP")
    
    with gr.Row():
        aa_decrement_button = gr.Button("- AA")
        aa_input = gr.Number(label="AA Level", value=0)
        aa_increment_button = gr.Button("+ AA")
    
    auto_optimize_button = gr.Button("Optimize")
    points_input = gr.Number(label="Points Available", value=50)
    total_invested_points = gr.Number(label="Total Invested Points", value=0, interactive=False)
    summary_output = gr.Dataframe()
    table_output = gr.Dataframe()
    
   
    graph_dps_curve = gr.Plot()
    
    points_input.change(set_available_points, inputs=points_input)
    initialize_button.click(initialize_table, inputs=[base_input, crit_input, critmin_input, critmax_input, points_input], outputs=[summary_output, table_output, mom_input, wp_input, aa_input, total_invested_points, points_input])
    
    mom_increment_button.click(update_mom, inputs=None, outputs=[summary_output, table_output, mom_input, total_invested_points])
    wp_increment_button.click(update_wp, inputs=None, outputs=[summary_output, table_output, wp_input, total_invested_points])
    aa_increment_button.click(update_aa, inputs=None, outputs=[summary_output, table_output, aa_input, total_invested_points])
    
    mom_decrement_button.click(decrement_mom, inputs=None, outputs=[summary_output, table_output, mom_input, total_invested_points])
    wp_decrement_button.click(decrement_wp, inputs=None, outputs=[summary_output, table_output, wp_input, total_invested_points])
    aa_decrement_button.click(decrement_aa, inputs=None, outputs=[summary_output, table_output, aa_input, total_invested_points])
    
    mom_input.change(update_mom, inputs=mom_input, outputs=[summary_output, table_output, mom_input, total_invested_points])
    wp_input.change(update_wp, inputs=wp_input, outputs=[summary_output, table_output, wp_input, total_invested_points])
    aa_input.change(update_aa, inputs=aa_input, outputs=[summary_output, table_output, aa_input, total_invested_points])
    auto_optimize_button.click(automatic_optimization, inputs=points_input, outputs=[summary_output, table_output, mom_input, wp_input, aa_input, total_invested_points, points_input])
    
   
    points_input.change(plot_dps_curve, inputs=None, outputs=graph_dps_curve)

demo.launch()
