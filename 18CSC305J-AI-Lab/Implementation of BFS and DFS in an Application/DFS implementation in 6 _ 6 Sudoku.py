def print_sudoku(arr):
	for i in range(len(arr)):
		for j in range(len(arr[0])):
			print(str(arr[i][j])+" ",end="")
		print ('')


def empty_locs(arr, l):
	for row in range(6):
		for col in range(6):
			if(arr[row][col]== 0):
				l[0]= row
				l[1]= col
				return True
	return False

def used_in_row(arr, row, num):
	for i in range(6):
		if(arr[row][i] == num):
			return True
	return False

def used_in_col(arr, col, num):
	for i in range(6):
		if(arr[i][col] == num):
			return True
	return False

def used_in_box(arr, row, col, num):
	for i in range(2):
		for j in range(3):
			if(arr[i + row][j + col] == num):
				return True
	return False

def check_location_is_safe(arr, row, col, num):
	
	
	return not used_in_row(arr, row, num) and not used_in_col(arr, col, num) and not used_in_box(arr, row - row % 2,
						col - col % 3, num)


def solver(arr):
	

	l =[0, 0]

	if(not empty_locs(arr, l)):
		return True

	row = l[0]
	col = l[1]
	
	for num in range(1,7):
		
		if(check_location_is_safe(arr,
						row, col, num)):
			
			arr[row][col]= num

			if(solver(arr)):
				return True

			arr[row][col] = 0
			
	return False

if __name__=="__main__":
	
	grid =[[0 for x in range(6)]for y in range(6)]
	
	grid = [[0,0,0,1,0,6],
            [6,0,3,0,0,0],
            [1,0,2,0,4,0],
            [0,0,0,5,0,1],
            [0,1,0,6,0,6],
            [5,0,6,0,0,4]]
	
	if(solver(grid)):
		print_sudoku(grid)
	else:
		print("No solution exists")
