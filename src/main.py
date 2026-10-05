#!/usr/bin/env python3
import sys


def read_file_lines(path: str) -> list[bytes]:
    """Reads a file as raw bytes and splits into lines according to assignment rules:

    1. Read the file as raw bytes, not as text.
    2. Split the content on the newline byte \n.
    3. If the last piece is empty, drop it.
    4. Keep any \r as part of the line.
    
    """
    try:
        with open(path, "rb") as f:
            content = f.read()
    except OSError as e:
        sys.stderr.write(f"Error: Unable to read file '{path}': {e}\n")
        sys.exit(2)

    if not content:
        return []

    lines = content.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()

    return lines


def _myers_core(A: list, B: list) -> list[tuple[str, any]]:
    """Core O(ND) Myers diff algorithm with compact history allocation."""
    N = len(A)
    M = len(B)

    # Map elements to unique integer IDs for fast equality comparison in snakes
    item_ids = {}
    id_A = [item_ids.setdefault(item, len(item_ids)) for item in A]
    id_B = [item_ids.setdefault(item, len(item_ids)) for item in B]

    history = []

    # d = 0: follow the initial diagonal snake from (0, 0)
    x = 0
    y = 0
    while x < N and y < M and id_A[x] == id_B[y]:
        x += 1
        y += 1
    history.append([x])

    if x >= N and y >= M:
        return [(" ", item) for item in A]

    d = 0
    found = False
    while not found:
        d += 1
        prev = history[d - 1]
        curr = [0] * (d + 1)

        # Diagonals k = -d, -d+2, ..., d
        # i maps 0 .. d to diagonal k = -d + 2*i
        for i in range(d + 1):
            k = -d + 2 * i

            if i == 0:
                # Came from k + 1 (vertical move: insertion)
                x = prev[0]
            elif i == d:
                # Came from k - 1 (horizontal move: deletion)
                x = prev[d - 1] + 1
            else:
                p_left = prev[i - 1]  # from k - 1
                p_right = prev[i]      # from k + 1
                x = p_right if p_left < p_right else p_left + 1

            y = x - k

            # Snake: greedily advance along matching diagonals
            while x < N and y < M and id_A[x] == id_B[y]:
                x += 1
                y += 1

            curr[i] = x

            if x >= N and y >= M:
                found = True
                history.append(curr)
                break

        if not found:
            history.append(curr)

    # Backtrack from (N, M) to (0, 0) using recorded history
    script = []
    curr_x = N
    curr_y = M

    for step_d in range(d, 0, -1):
        k = curr_x - curr_y
        i = (k + step_d) // 2
        prev = history[step_d - 1]

        if i == 0:
            prev_k = k + 1
            prev_i = 0
        elif i == step_d:
            prev_k = k - 1
            prev_i = step_d - 1
        else:
            if prev[i - 1] < prev[i]:
                prev_k = k + 1
                prev_i = i
            else:
                prev_k = k - 1
                prev_i = i - 1

        prev_x = prev[prev_i]
        prev_y = prev_x - prev_k

        if prev_k == k - 1:
            # Horizontal move: Deletion of A[prev_x]
            # Diagonal snake follows from (prev_x + 1, prev_y) to (curr_x, curr_y)
            snake_len = curr_x - (prev_x + 1)
            for s in range(snake_len - 1, -1, -1):
                script.append((" ", A[prev_x + 1 + s]))
            script.append(("-", A[prev_x]))
        else:
            # Vertical move: Insertion of B[prev_y]
            # Diagonal snake follows from (prev_x, prev_y + 1) to (curr_x, curr_y)
            snake_len = curr_x - prev_x
            for s in range(snake_len - 1, -1, -1):
                script.append((" ", B[prev_y + 1 + s]))
            script.append(("+", B[prev_y]))

        curr_x = prev_x
        curr_y = prev_y

    # Diagonal snake for d = 0 from (0, 0) to curr_x
    for s in range(curr_x - 1, -1, -1):
        script.append((" ", A[s]))

    script.reverse()
    return script


def myers_diff(A: list, B: list) -> list[tuple[str, any]]:
    """Runs Myers diff with O(N) common prefix and suffix trimming."""
    N = len(A)
    M = len(B)

    # 1. Strip common prefix
    p = 0
    min_len = min(N, M)
    while p < min_len and A[p] == B[p]:
        p += 1

    # 2. Strip common suffix
    s = 0
    while s < min_len - p and A[N - 1 - s] == B[M - 1 - s]:
        s += 1

    prefix_script = [(" ", A[i]) for i in range(p)]
    suffix_script = [(" ", A[N - s + i]) for i in range(s)]

    mid_A = A[p : N - s]
    mid_B = B[p : M - s]

    if not mid_A and not mid_B:
        return prefix_script + suffix_script
    if not mid_A:
        return prefix_script + [("+", item) for item in mid_B] + suffix_script
    if not mid_B:
        return prefix_script + [("-", item) for item in mid_A] + suffix_script

    mid_script = _myers_core(mid_A, mid_B)
    return prefix_script + mid_script + suffix_script


def format_ranges(indices: list[int]) -> str:
    """Formats 0-indexed Unicode code point indices into merged half-open ranges [start, end)."""
    if not indices:
        return "."

    ranges = []
    start = indices[0]
    end = indices[0] + 1

    for idx in indices[1:]:
        if idx == end:
            end += 1
        else:
            ranges.append(f"{start}-{end}")
            start = idx
            end = idx + 1

    ranges.append(f"{start}-{end}")
    return ",".join(ranges)


def get_highlight_line(old_str: str, new_str: str) -> str:
    """Computes changed Unicode character ranges between a paired old line and new line."""
    char_script = myers_diff(list(old_str), list(new_str))
    idx_a = 0
    idx_b = 0
    del_a = []
    ins_b = []

    for op, _ in char_script:
        if op == "-":
            del_a.append(idx_a)
            idx_a += 1
        elif op == "+":
            ins_b.append(idx_b)
            idx_b += 1
        else:
            idx_a += 1
            idx_b += 1

    return f"? {format_ranges(del_a)} | {format_ranges(ins_b)}"


def run_diff(mode: str, path_a: str, path_b: str) -> None:
    A = read_file_lines(path_a)
    B = read_file_lines(path_b)

    script = myers_diff(A, B)

    # Output lines following the delete-first rule within each change block
    out = []
    i = 0
    n = len(script)

    while i < n:
        op, val = script[i]
        if op == " ":
            out.append(b" " + val + b"\n")
            i += 1
        else:
            # Change block: collect all consecutive non-keep lines
            minus_lines = []
            plus_lines = []
            while i < n and script[i][0] != " ":
                if script[i][0] == "-":
                    minus_lines.append(script[i][1])
                else:
                    plus_lines.append(script[i][1])
                i += 1

            # Delete-first rule: print all - lines before any + line
            for line in minus_lines:
                out.append(b"-" + line + b"\n")

            for j, line in enumerate(plus_lines):
                out.append(b"+" + line + b"\n")
                if mode == "highlight" and j < len(minus_lines):
                    # Paired with minus_lines[j]
                    old_str = minus_lines[j].decode("utf-8")
                    new_str = line.decode("utf-8")
                    hl = get_highlight_line(old_str, new_str)
                    out.append(hl.encode("utf-8") + b"\n")

    sys.stdout.buffer.write(b"".join(out))


def main():
    if len(sys.argv) != 4 or sys.argv[1] not in ("lines", "highlight"):
        sys.stderr.write("Usage: python3 src/main.py <lines|highlight> <file_A> <file_B>\n")
        sys.exit(2)

    command = sys.argv[1]
    file_a = sys.argv[2]
    file_b = sys.argv[3]

    run_diff(command, file_a, file_b)


if __name__ == "__main__":
    main()