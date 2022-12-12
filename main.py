# libs

# code
import randomwalk



def main():
    grid, coord_vec = randomwalk.random_walk(5, 100)
    print(grid)
    for coord in coord_vec:
        print(coord.x, coord.y, coord.amin)
    return

if __name__=="__main__":
    main()