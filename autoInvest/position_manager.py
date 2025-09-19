import json

def save_positions(positions, filename):
    """
    매수/매도 포지션 리스트를 JSON 파일로 저장

    :param positions: 포지션 리스트
    :param filename: 저장 파일명
    """
    with open(filename, 'w') as f:
        json.dump(positions, f)

def load_positions(filename):
    """
    저장된 포지션 JSON 파일을 불러옴

    :param filename: 불러올 파일명
    :return: 포지션 리스트
    """
    with open(filename, 'r') as f:
        return json.load(f)