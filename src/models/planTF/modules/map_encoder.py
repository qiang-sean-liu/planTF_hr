import torch
import torch.nn as nn

from ..layers.embedding import PointsEncoder


class MapEncoder(nn.Module):
    def __init__(
        self,
        polygon_channel=6,
        dim=128,
        remove_global_pos=True,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.polygon_encoder = PointsEncoder(polygon_channel, dim)
        self.speed_limit_emb = nn.Sequential(
            nn.Linear(1, dim), nn.ReLU(), nn.Linear(dim, dim)
        )

        self.type_emb = nn.Embedding(3, dim)
        self.on_route_emb = nn.Embedding(2, dim)
        self.traffic_light_emb = nn.Embedding(4, dim)
        self.unknown_speed_emb = nn.Embedding(1, dim)

        self.remove_global_pos = remove_global_pos

    def forward(self, data) -> torch.Tensor:
        polygon_center = data["map"]["polygon_center"]  # (bs, M, 3), where M is number of polygons, "3" is for "x", "y", "theta"
        polygon_type = data["map"]["polygon_type"].long()  # (bs, M)
        polygon_on_route = data["map"]["polygon_on_route"].long()  # (bs, M)
        polygon_tl_status = data["map"]["polygon_tl_status"].long()  # (bs, M)
        polygon_has_speed_limit = data["map"]["polygon_has_speed_limit"]  # (bs, M)
        polygon_speed_limit = data["map"]["polygon_speed_limit"]  # (bs, M)
        point_position = data["map"]["point_position"]  # (bs, M, 3, P, 2), where "3" is for "center", "left", "right", and "P" is number of points of each polygon
        point_vector = data["map"]["point_vector"]  # (bs, M, 3, P, 2)
        point_orientation = data["map"]["point_orientation"]  # (bs, M, 3, P)
        valid_mask = data["map"]["valid_mask"]  # (bs, M, P)

        if self.remove_global_pos:
            pt_pos = point_position[:, :, 0] - polygon_center[..., None, :2]  # (bs, M, P, 2). NOTE: Left and right lanelines are NOT used ???
        else:
            pt_pos = point_position[:, :, 0]  # (bs, M, P, 2)

        polygon_feature = torch.cat(
            [
                pt_pos,
                point_vector[:, :, 0],
                torch.stack(
                    [
                        point_orientation[:, :, 0].cos(),
                        point_orientation[:, :, 0].sin(),
                    ],
                    dim=-1,
                ),
            ],
            dim=-1,
        )

        bs, M, P, C = polygon_feature.shape
        valid_mask = valid_mask.view(bs * M, P)
        polygon_feature = polygon_feature.reshape(bs * M, P, C)

        x_polygon = self.polygon_encoder(polygon_feature, valid_mask).view(bs, M, -1)

        x_type = self.type_emb(polygon_type)
        x_on_route = self.on_route_emb(polygon_on_route)
        x_tl_status = self.traffic_light_emb(polygon_tl_status)
        x_speed_limit = torch.zeros(bs, M, self.dim, device=x_polygon.device)
        x_speed_limit[polygon_has_speed_limit] = self.speed_limit_emb(
            polygon_speed_limit[polygon_has_speed_limit].unsqueeze(-1)
        )
        x_speed_limit[~polygon_has_speed_limit] = self.unknown_speed_emb.weight

        x_polygon += x_type + x_on_route + x_tl_status + x_speed_limit

        return x_polygon
