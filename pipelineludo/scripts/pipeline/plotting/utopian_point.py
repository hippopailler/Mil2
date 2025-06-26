class Point:
    """
    Class to represent a point in a plot.

    :param x: x-coordinate of the point.
    :param y: y-coordinate of the point.
    :param color: Color of the point.
    :param attributes: Attributes of the point.
    """
    color = ''
    attributes = {}

    def __init__(self, x: float, y: float, color: str = 'red', attributes: dict = {}):
        self.x = x
        self.y = y
        self.color = color
        self.attributes = attributes


class Line:
    """
    Class to represent a line in a plot.

    :param x_values: x-coordinates of the line.
    :param y_values: y-coordinates of the line.
    :param color: Color of the line.
    :param attributes: Attributes of the line.
    """
    color = ''
    attributes = {}

    def __init__(self, x_values: list[float], y_values: list[float], color: str = '#636EFA', attributes: dict = {}):
        self.x_values = x_values
        self.y_values = y_values
        self.color = color
        self.attributes = attributes


y_train = np.array([0, 1, 0, 1, 0, 1])  # Example y_train data
all_false_negative_perc_cv_mean = np.array([0.1, 0.2, 0.3, 0.34, 0.4])  # Example false negative percentages
all_negative_predicted_percentage_cv_mean = np.array([0.4, 0.5, 0.6, 0.62, 0.65])  # Example negative predicted percentages
model_data_cv = {
    'F1': [0.6, 0.67, 0.57, 0.64, 0.7],
    'Precision': [0.65, 0.7, 0.6, 0.68, 0.72],
    'Recall': [0.55, 0.6, 0.54, 0.6, 0.66],
    'AUC': [0.7, 0.75, 0.68, 0.72, 0.78],
}  # Example model data

plot_data_cv = {
    'points': {
        'Utopian point': Point(0, (np.sum(y_train == 0) / y_train.size), 'red'),
    },
    'lines': {
        'LUDOLE': Line(
            all_false_negative_perc_cv_mean,
            all_negative_predicted_percentage_cv_mean,
            '#636EFA',
            model_data_cv
        )
    },
    'graph_title': 'Cost search',
    'x_axis_name': 'False negative percentage',
    'y_axis_name': 'Negative predicted percentage'
}

def plot_cost_search(data: dict) -> go.Figure:
    points: dict[str: Point] = data['points']
    lines: dict[str: Line] = data['lines']

    fig = go.Figure()

    for point_name, point_data in points.items():
        fig.add_trace(go.Scatter(
            x=[point_data.x],
            y=[point_data.y],
            mode='markers',
            marker=dict(color=point_data.color),
            name=point_name)
        )

    for line_name, line_data in lines.items():
        hover_text = [
            '<br>'.join(f'{key}: {value}' for key, value in zip(line_data.attributes.keys(), values))
            for values in zip(*line_data.attributes.values())
        ]

        fig.add_trace(go.Scatter(
            x=line_data.x_values,
            y=line_data.y_values,
            mode='lines+markers',
            marker=dict(color=line_data.color),
            text=hover_text,
            name=line_name,
            hovertemplate='%{text}'
        ))

    fig.update_layout(title=data['graph_title'],
                    xaxis_title=data['x_axis_name'],
                    yaxis_title=data['y_axis_name'])

    return fig

plot_cost_search(plot_data_cv)

def distance_from_utopia_negative(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    utopian_point = [0, (np.sum(y_true == 0) / len(y_true))]

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    x = fn / (tn + fn + tp + fp)
    y = (tn + fn) / (tn + fn + tp + fp)

    distance = np.sqrt((x - utopian_point[0]) ** 2 + (y - utopian_point[1]) ** 2)

    return distance