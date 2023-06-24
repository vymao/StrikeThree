import pitcher
from pprint import pprint

def main():
    p = pitcher.PitcherEnv()
    print(len(p.get_data()))
    print(p.get_data())
    print(p.releasepoint_reward)

if __name__ == "__main__":
    main()
