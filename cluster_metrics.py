import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json


def build_average_grade_by_year(data):

    """Построение графика средней оценки по годам."""

    average_grade_by_year = data.groupby('year')['score'].mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=average_grade_by_year.index,
        y=average_grade_by_year.values,
        mode='lines+markers',
        marker=dict(color='white'),
        line=dict(color='lightblue', width=5)
    ))
    fig.update_layout(
        title=dict(
            text='Средняя оценка по годам',
            font=dict(size=24),  
            x=0.5  
        ),
        xaxis_title='Год',
        yaxis_title='Средняя оценка',
        paper_bgcolor='rgb(87, 74, 87)',
        plot_bgcolor='rgb(87, 74, 87)',
        font=dict(color='white'),
        height=600
    )
    return fig.to_json()


def build_top_lecturers(data):

    """Построение графика средних оценок по преподавателям"""

    top_lecturers = data.groupby('advisor')['score'].mean()
    fig = px.bar(
        x=top_lecturers.index,
        y=top_lecturers.values,
        labels={'x': 'Преподаватель', 'y': 'Средняя оценка'},
        title='Средняя оценка по преподавателям'
    )
    fig.update_traces(marker_color='lightblue')
    fig.update_layout(
        title=dict(
            text='Средняя оценка по преподавателям',
            font=dict(size=24),  
            x=0.5  
        ),
        paper_bgcolor='rgb(87, 74, 87)',
        plot_bgcolor='rgb(87, 74, 87)',
        font=dict(color='white'),
        xaxis=dict(tickangle=45, showgrid=True)
    )
    return fig.to_json()


def build_count_per_year(data):

    """Построение графика количества работ по годам."""

    count_per_year = data.groupby('year')['name'].count()
    fig = px.bar(
        x=count_per_year.index,
        y=count_per_year.values,
        labels={'x': 'Год', 'y': 'Количество работ'},
        title='Количество работ по годам',
        height=600
    )
    fig.update_traces(marker_color='lightblue')
    fig.update_layout(
        title=dict(
            text='Количество работ по годам',
            font=dict(size=24),  
            x=0.5  
        ),
        paper_bgcolor='rgb(87, 74, 87)',
        plot_bgcolor='rgb(87, 74, 87)',
        font=dict(color='white'),
        xaxis=dict(showgrid=True)
    )
    return fig.to_json()


def build_top_popular_lecturers(data):
    """Построение графика топа популярных преподавателей."""
    top11_popular_lecturers = data.groupby('advisor')['name'].count().sort_values(ascending=False)[:11]
    fig = px.bar(
        x=top11_popular_lecturers.index,
        y=top11_popular_lecturers.values,
        labels={'x': 'Преподаватель', 'y': 'Количество работ'},
        title='Топ самых популярных преподавателей',
        height=600
    )
    fig.update_traces(marker_color='lightblue')
    fig.update_layout(
        title=dict(
            text='Топ самых популярных преподавателей',
            font=dict(size=24),  
            x=0.5  
        ),
        paper_bgcolor='rgb(87, 74, 87)',
        plot_bgcolor='rgb(87, 74, 87)',
        font=dict(color='white'),
        xaxis=dict(showgrid=True, tickangle=50, tickfont=dict(size=14))
    )
    return fig.to_json()


def generate_figures(data_path):
    """
    Основная функция, вызывающая построение всех графиков.
    """
    
    data = pd.read_json(data_path)
    data = data.dropna(subset=['score'])

    return {
        "average_grade_by_year": build_average_grade_by_year(data),
        "top_lecturers": build_top_lecturers(data),
        "count_per_year": build_count_per_year(data),
        "top_popular_lecturers": build_top_popular_lecturers(data)
    }


if __name__ == "__main__":
    
    data_path = 'decoded_data_fcl_filtered.json'

    
    figures = generate_figures(data_path)

    
    for name, figure_json in figures.items():
        with open(f"{name}.json", "w") as f:
            f.write(figure_json)

