function [angles, side_lengths] = triangle_calculations(p1, p2, p3)
% triangle_calculations: Calculates angles and side lengths of a triangle.
%
%   [angles, side_lengths] = triangle_calculations(p1, p2, p3)
%
%   Inputs:
%       p1, p2, p3: 1x2 or 1x3 vectors representing the coordinates of the
%                   three vertices of the triangle.
%
%   Outputs:
%       angles:     3x1 vector containing the angles of the triangle in degrees.
%       side_lengths: 3x1 vector containing the lengths of the three sides.

    % Calculate side lengths
    a = norm(p2 - p3);
    b = norm(p1 - p3);
    c = norm(p1 - p2);
    side_lengths = [a; b; c];

    % Calculate angles using the Law of Cosines
    angle_a = acosd((b^2 + c^2 - a^2) / (2 * b * c));
    angle_b = acosd((a^2 + c^2 - b^2) / (2 * a * c));
    angle_c = acosd((a^2 + b^2 - c^2) / (2 * a * b));
    angles = [angle_a; angle_b; angle_c];
end
