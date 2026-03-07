"""Модуль графовых нейронных сетей (GNN) на основе KokaoCore."""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Union
import logging
from pathlib import Path
from dataclasses import dataclass, field

from .core import KokaoCore
from .core_base import CoreConfig

logger = logging.getLogger(__name__)


@dataclass
class Graph:
    """
    Структура графа для GNN.
    """
    num_nodes: int
    edge_index: torch.Tensor  # (2, num_edges)
    node_features: Optional[torch.Tensor] = None  # (num_nodes, node_dim)
    edge_features: Optional[torch.Tensor] = None  # (num_edges, edge_dim)
    graph_features: Optional[torch.Tensor] = None

    def __post_init__(self):
        if self.node_features is not None:
            assert len(self.node_features) == self.num_nodes
        assert self.edge_index.shape[0] == 2


class GraphConvolution(nn.Module):
    """
    Слой графовой свертки (GCN-style).
    Использует KokaoCore для агрегации сообщений от соседей.
    """

    def __init__(self, in_features: int, out_features: int,
                 num_cores: int = 4, activation: str = 'relu'):
        """
        Инициализация слоя свертки.

        Args:
            in_features: Размерность входа
            out_features: Размерность выхода
            num_cores: Количество ядер KokaoCore
            activation: Функция активации
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_cores = num_cores

        # Преобразование признаков
        self.linear = nn.Linear(in_features, out_features)

        # Ядра для агрегации
        self.cores = nn.ModuleList([
            KokaoCore(CoreConfig(input_dim=in_features))
            for _ in range(num_cores)
        ])

        # Выходное преобразование
        self.output_transform = nn.Linear(num_cores, out_features)

        # Активация
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Identity()

    def forward(self, x: torch.Tensor, 
                edge_index: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход свертки.

        Args:
            x: Признаки узлов (num_nodes, in_features)
            edge_index: Индексы ребер (2, num_edges)

        Returns:
            Обновленные признаки узлов (num_nodes, out_features)
        """
        num_nodes = len(x)

        # Линейное преобразование
        h = self.linear(x)

        # Агрегация сообщений от соседей
        messages = []
        for core in self.cores:
            aggregated = self._aggregate(x, edge_index, core, num_nodes)
            messages.append(aggregated)

        # Конкатенация и преобразование
        messages = torch.stack(messages, dim=-1)  # (num_nodes, out_features, num_cores)
        output = self.output_transform(messages)
        output = self.activation(output)

        return output

    def _aggregate(self, x: torch.Tensor, edge_index: torch.Tensor,
                   core: KokaoCore, num_nodes: int) -> torch.Tensor:
        """
        Агрегация сообщений от соседей с использованием KokaoCore.

        Args:
            x: Признаки узлов
            edge_index: Индексы ребер
            core: KokaoCore для агрегации
            num_nodes: Количество узлов

        Returns:
            Агрегированные сообщения
        """
        aggregated = torch.zeros(num_nodes, self.out_features, device=x.device)

        # Для каждого узла собираем сообщения от соседей
        source, target = edge_index[0], edge_index[1]

        for i in range(num_nodes):
            # Индексы соседей
            neighbor_mask = target == i
            neighbor_indices = source[neighbor_mask]

            if len(neighbor_indices) > 0:
                # Признаки соседей
                neighbor_features = x[neighbor_indices]

                # Агрегация через KokaoCore
                signals = []
                for neighbor_feat in neighbor_features:
                    signal = core.signal(neighbor_feat)
                    signals.append(signal)

                # Средний сигнал
                avg_signal = torch.mean(torch.tensor(signals))
                aggregated[i] = avg_signal * x[i][:self.out_features]

        return aggregated


class GraphAttentionLayer(nn.Module):
    """
    Слой графового внимания (GAT-style).
    """

    def __init__(self, in_features: int, out_features: int,
                 num_heads: int = 4, dropout: float = 0.1):
        """
        Инициализация слоя внимания.

        Args:
            in_features: Размерность входа
            out_features: Размерность выхода
            num_heads: Количество голов внимания
            dropout: Dropout rate
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads

        # Линейные преобразования для каждой головы
        self.linear_q = nn.ModuleList([
            nn.Linear(in_features, self.head_dim) for _ in range(num_heads)
        ])
        self.linear_k = nn.ModuleList([
            nn.Linear(in_features, self.head_dim) for _ in range(num_heads)
        ])
        self.linear_v = nn.ModuleList([
            nn.Linear(in_features, self.head_dim) for _ in range(num_heads)
        ])

        # Выходное преобразование
        self.output_transform = nn.Linear(out_features, out_features)

        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor,
                return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Прямой проход внимания.

        Args:
            x: Признаки узлов
            edge_index: Индексы ребер
            return_attention: Вернуть ли веса внимания

        Returns:
            Обновленные признаки узлов
        """
        num_nodes = len(x)
        outputs = []
        attention_weights = []

        source, target = edge_index[0], edge_index[1]

        for head in range(self.num_heads):
            # Проекции
            q = self.linear_q[head](x)  # (num_nodes, head_dim)
            k = self.linear_k[head](x)
            v = self.linear_v[head](x)

            # Вычисление внимания для каждого ребра
            edge_attention = []
            for i in range(len(source)):
                s, t = source[i].item(), target[i].item()
                attention_score = torch.dot(q[t], k[s])
                edge_attention.append(attention_score)

            edge_attention = torch.tensor(edge_attention)

            # Softmax по входящим ребрам для каждого узла
            attention_per_node = []
            for node in range(num_nodes):
                node_mask = target == node
                node_attentions = edge_attention[node_mask]
                if len(node_attentions) > 0:
                    node_weights = self.softmax(node_attentions)
                else:
                    node_weights = torch.tensor([])
                attention_per_node.append(node_weights)

            # Агрегация значений
            aggregated = torch.zeros(num_nodes, self.head_dim, device=x.device)
            for node in range(num_nodes):
                node_mask = target == node
                neighbor_indices = source[node_mask]
                if len(neighbor_indices) > 0 and len(attention_per_node[node]) > 0:
                    weights = attention_per_node[node].unsqueeze(-1)
                    values = v[neighbor_indices]
                    aggregated[node] = (weights * values).sum(dim=0)

            outputs.append(aggregated)

        # Конкатенация голов
        output = torch.cat(outputs, dim=-1)
        output = self.output_transform(output)

        if return_attention:
            return output, torch.cat(attention_per_node) if attention_per_node else None
        return output


class KokaoGNN(nn.Module):
    """
    Графовая нейронная сеть на основе KokaoCore.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 output_dim: int = 1, num_layers: int = 3,
                 num_heads: int = 4, dropout: float = 0.1):
        """
        Инициализация GNN.

        Args:
            input_dim: Размерность входа
            hidden_dim: Размерность скрытого слоя
            output_dim: Размерность выхода
            num_layers: Количество слоев
            num_heads: Количество голов внимания
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # Входное преобразование
        self.input_transform = nn.Linear(input_dim, hidden_dim)

        # Слои GAT
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim if i > 0 else hidden_dim
            self.layers.append(
                GraphAttentionLayer(in_dim, hidden_dim, num_heads, dropout)
            )

        # Выходной слой
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, graph: Graph) -> torch.Tensor:
        """
        Прямой проход GNN.

        Args:
            graph: Граф для обработки

        Returns:
            Признаки узлов или графа
        """
        x = graph.node_features
        if x is None:
            # Если нет признаков, используем one-hot encoding узлов
            x = torch.eye(graph.num_nodes)
            x = self.input_transform(x)
        else:
            x = self.relu(self.input_transform(x))

        # Применение слоев GNN
        for layer in self.layers:
            x = layer(x, graph.edge_index)
            x = self.dropout(x)
            x = self.relu(x)

        # Выход
        output = self.output_layer(x)

        return output

    def predict_graph_level(self, graph: Graph) -> torch.Tensor:
        """
        Предсказание на уровне графа (global pooling).

        Args:
            graph: Граф для обработки

        Returns:
            Предсказание для всего графа
        """
        node_outputs = self.forward(graph)

        # Global mean pooling
        graph_output = node_outputs.mean(dim=0, keepdim=True)

        return graph_output


class GraphDataset:
    """
    Датасет для графовых данных.
    """

    def __init__(self, graphs: List[Graph], 
                 labels: Optional[List[float]] = None):
        """
        Инициализация датасета.

        Args:
            graphs: Список графов
            labels: Метки для каждого графа
        """
        self.graphs = graphs
        self.labels = labels

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Tuple[Graph, Optional[float]]:
        label = self.labels[idx] if self.labels else None
        return self.graphs[idx], label

    def create_batch(self, indices: List[int]) -> Tuple[List[Graph], Optional[torch.Tensor]]:
        """
        Создание батча из графов.

        Args:
            indices: Индексы графов для батча

        Returns:
            (список графов, метки)
        """
        batch_graphs = [self.graphs[i] for i in indices]
        batch_labels = None
        if self.labels:
            batch_labels = torch.tensor([self.labels[i] for i in indices])

        return batch_graphs, batch_labels


def create_random_graph(num_nodes: int, edge_probability: float = 0.3,
                        node_features_dim: int = 16) -> Graph:
    """
    Создание случайного графа.

    Args:
        num_nodes: Количество узлов
        edge_probability: Вероятность наличия ребра
        node_features_dim: Размерность признаков узлов

    Returns:
        Случайный граф
    """
    # Генерация ребер
    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.random.random() < edge_probability:
                edges.append([i, j])
                edges.append([j, i])  # Неориентированный граф

    if not edges:
        # Добавить хотя бы одно ребро
        edges = [[0, 1], [1, 0]]

    edge_index = torch.tensor(edges, dtype=torch.long).t()

    # Признаки узлов
    node_features = torch.randn(num_nodes, node_features_dim)

    return Graph(
        num_nodes=num_nodes,
        edge_index=edge_index,
        node_features=node_features
    )


def karate_club_graph() -> Graph:
    """
    Загрузка графа каратного клуба (Zachary's Karate Club).

    Returns:
        Граф каратного клуба
    """
    # Ребра графа
    edges = [
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8),
        (0, 10), (0, 11), (0, 12), (0, 13), (0, 17), (0, 19), (0, 21), (0, 31),
        (1, 2), (1, 3), (1, 7), (1, 13), (1, 17), (1, 19), (1, 21), (1, 30),
        (2, 3), (2, 7), (2, 8), (2, 9), (2, 13), (2, 27), (2, 28), (2, 32),
        (3, 7), (3, 12), (3, 13),
        (4, 6), (4, 10),
        (5, 6), (5, 10), (5, 16),
        (6, 16),
        (8, 30), (8, 32), (8, 33),
        (9, 33),
        (13, 33),
        (14, 32), (14, 33),
        (15, 32), (15, 33),
        (18, 32), (18, 33),
        (19, 33),
        (20, 32), (20, 33),
        (22, 32), (22, 33),
        (23, 25), (23, 27), (23, 29), (23, 32), (23, 33),
        (24, 25), (24, 27), (24, 31),
        (25, 31),
        (26, 29), (26, 33),
        (27, 33),
        (28, 31), (28, 33),
        (29, 32), (29, 33),
        (30, 32), (30, 33),
        (31, 32), (31, 33),
        (32, 33)
    ]

    num_nodes = 34
    edge_index = torch.tensor(
        [(i, j) for i, j in edges] + [(j, i) for i, j in edges],
        dtype=torch.long
    ).t()

    node_features = torch.randn(num_nodes, 16)

    return Graph(
        num_nodes=num_nodes,
        edge_index=edge_index,
        node_features=node_features
    )
