import pandas as pd
import json
import sys
import numpy as np

def process_scatter_plot(df):
    scatter_data = []
    for course, group in df.groupby("course"):
        scatter_data.append({
            "id": course,
            "data": [{"x": int(row["year"]), "y": int(row["score"])} for _, row in group.iterrows()]
        })
    return scatter_data

def process_line_graph(df):
    average_scores_by_year = df.groupby('year')['score'].mean().reset_index()
    return [
        {
            "id": "Средний балл по годам",
            "data": [{"x": int(row['year']), "y": row['score']} for _, row in average_scores_by_year.iterrows()]
        }
    ]

def generate_graph_json(df, main_param, related_param):
    main_values = df[main_param].unique()
    related_values = df[related_param].unique()

    nodes = []
    for value in main_values:
        nodes.append({
            "id": int(value) if main_param in ['year', 'score'] else value,
            "height": 1,
            "size": 24,
            "color": "rgb(97, 205, 187)"
        })
    
    for value in related_values:
        nodes.append({
            "id": int(value) if related_param in ['year', 'score'] else value,
            "height": 0,
            "size": 12,
            "color": "rgb(232, 193, 160)"
        })

    links = []
    for _, row in df.iterrows():
        links.append({
            "source": int(row[main_param]) if main_param in ['year', 'score'] else row[main_param],
            "target": int(row[related_param]) if related_param in ['year', 'score'] else row[related_param]
        })
    
    return {"nodes": nodes, "links": links}

def create_grouped_json(df, group_col):
    grouped_counts = df[group_col].value_counts().sort_index().reset_index()
    grouped_counts.columns = ['id', 'value']
    return grouped_counts.to_dict(orient='records')

def main(df):
    

    df['score'] = df['score'].astype(int)
    df['year'] = df['year'].astype(int)

    
    scatter_plot_data = process_scatter_plot(df)
    line_graph_data = process_line_graph(df)
    json_author_name = generate_graph_json(df, 'author', 'name')
    json_score_author = generate_graph_json(df, 'score', 'author')
    json_advisor_name = generate_graph_json(df, 'advisor', 'name')
    json_score_name = generate_graph_json(df, 'score', 'name')
    json_year_advisor = generate_graph_json(df, 'year', 'advisor')
    json_year_name = generate_graph_json(df, 'year', 'name')
    grouped_json_year = create_grouped_json(df, 'year')
    grouped_json_score = create_grouped_json(df, 'score')

    
    return {
        "scatter_plot": scatter_plot_data,
        "line_graph": line_graph_data,
        "author_name_graph": json_author_name,
        "score_author_graph": json_score_author,
        "advisor_name_graph": json_advisor_name,
        "score_name_graph": json_score_name,
        "year_advisor_graph": json_year_advisor,
        "year_name_graph": json_year_name,
        "grouped_by_year": grouped_json_year,
        "grouped_by_score": grouped_json_score
    }

if __name__ == "__main__":
    
    input_data = pd.read_json(sys.argv[1])

    
    output_data = main(input_data)

    
    print(json.dumps(output_data, ensure_ascii=False, indent=4))
