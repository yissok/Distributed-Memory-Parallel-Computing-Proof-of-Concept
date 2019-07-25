#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <unistd.h>

#define tag1 42
#define tag2 24
#define dmnsn 10

#define DEBUG 0


void check_double_malloc(double* square_array)
{
    if (square_array == NULL)
    {
        fprintf(stderr, "No space for the array\n");
        exit(EXIT_FAILURE);
    }
}


//populates the array with doubles between 0 and 1
double* initialise_square_array(int dim)
{
    long unsigned int dimension = (long unsigned int) dim;
    double *sq_array;
    int i, j;
    sq_array = malloc(dimension * dimension * sizeof(double));
    check_double_malloc(sq_array);
    for (i = 0; i < dim; i++)
    {
        for (j = 0; j < dim; j++)
        {
            sq_array[i*dim+j] = (double)(rand() % 100)/100;
        }
    }
    return sq_array;
}


double* create_new_array(int dim_x, int dim_y)
{
    long unsigned int dimension_x = (long unsigned int) dim_x;
    long unsigned int dimension_y = (long unsigned int) dim_y;
    double *arr;
    arr = malloc(dimension_x * dimension_y * sizeof(double));
    check_double_malloc(arr);
    return arr;
}


void print_array(int dimension, double* square_array)
{
    int i, j;
    for (i = 0; i < dimension; i++)
    {
        for (j = 0; j < dimension; j++)
        {
            printf("%f ", square_array[i*dimension+j]);
        }
        printf("\n");
    }
}


void print_non_square_array(int y, int x, double* square_array,int rnk_num)
{
    int i, j;
    for (i = 0; i < y; i++)
    {
        printf("%d: ", rnk_num);
        for (j = 0; j < x; j++)
        {
            printf("%f ", square_array[i*x+j]);
        }
        printf("\n");
    }
}


/*
VITAL FUNCTION: given the array dimension, the total number of processes and
the index of a process, the function tells where that process should begin
*/
int get_range_from_p_num_start(int dimension, int number_of_processes,
                               int process_rank)
{
    float div= ((float)dimension)/number_of_processes;
    if(div<1)
    {
        printf("too many processes, too few rows to work on\n");
        return (-1);
    }
    int chunk_height = (int)floor(div);
    int remainder=dimension%number_of_processes;
    //the remainder will be split betweem the first n processes.
    //The last (dimension-n) processes will have 1 fewer row as a result
    int floor_rank_chunks=0;//counter for normal chunks
    int increased_rank_chunks=0;//counter for chunks with added reminder
    int start_height;
    if((process_rank-1)<remainder)
    {
        increased_rank_chunks=process_rank-1;
    }
    else
    {
        floor_rank_chunks=(process_rank-1)-remainder;
        increased_rank_chunks=remainder;
    }
    start_height = floor_rank_chunks*chunk_height+
        increased_rank_chunks*(chunk_height+1)+1;
    //start_height is the weighted product of both types of chunks
    //(the increased one is 1 unit "taller")
    if(DEBUG==1)
    {
        printf("div: %d/%d = %f ,",dimension,number_of_processes,div);
        printf("remainder: %d\n",remainder);
        printf("chunk_height: %d \n",chunk_height);
        printf("floor_rank_chunks: %d \n",floor_rank_chunks);
        printf("increased_rank_chunks: %d \n",increased_rank_chunks);
    }
    return start_height;
}


/*
VITAL FUNCTION: as in start, given dimension, the total number of processes and
the index of a process, the function tells where that process should end
*/
//being exactly the same as the previous function, refer to that for comments
int get_range_from_p_num_end(int dimension, int number_of_processes,
                             int process_rank)
{
    float div= ((float)dimension)/number_of_processes;
    if(div<1)
    {
        printf("too many processes, too few rows to work on\n");
        return (-1);
    }
    int chunk_height = (int)floor(div);
    int remainder=dimension%number_of_processes;
    int floor_rank_chunks=0;
    int increased_rank_chunks=0;
    int start_height;
    if((process_rank)<remainder)
    {
        increased_rank_chunks=process_rank;
    }
    else
    {
        floor_rank_chunks=(process_rank)-remainder;
        increased_rank_chunks=remainder;
    }
    start_height = floor_rank_chunks*chunk_height+
        increased_rank_chunks*(chunk_height+1);
    if(DEBUG==1)
    {
        printf("div: %d/%d = %f ,",dimension,number_of_processes,div);
        printf("remainder: %d\n",remainder);
        printf("chunk_height: %d \n",chunk_height);
        printf("floor_rank_chunks: %d \n",floor_rank_chunks);
        printf("increased_rank_chunks: %d \n",increased_rank_chunks);
    }
    return start_height;
}


//fits a chunk returned from a child into the final array
double* stitch_array(double* sq_array, double* chunk_array, int start,
                     int end, int dimension)
{
    int i, j;
    for (i = start+1; i < end; i++)
    {
        for (j = 1; j < dimension - 1; j++)
        {
            sq_array[i*dimension+j] = chunk_array[(i-start)*dimension+j];
        }
    }
    return sq_array;
}


//retrieves the requested partition from the initially generated array
double* select_chunk(int dimension, double* square_array,
                     int start_row, int n_rows_per_chunk)
{
    double* chunk_array = create_new_array(n_rows_per_chunk,dimension);
    int end_of_line=0;
    if((n_rows_per_chunk+start_row)>dimension)
    {
        end_of_line=dimension;
    }
    else
    {
        end_of_line=n_rows_per_chunk+start_row;
    }
    int i, j;
    for (i = start_row; i < end_of_line; i++)
    {
        for (j = 0; j < dimension; j++)
        {
            chunk_array[(i-start_row)*(dimension)+j] = 
                square_array[i*(dimension)+j];
        }
    }
    return chunk_array;
}


void check_ptr_1D_mem(double **allocation)
{
    if (allocation == NULL)
    {
        printf("NO MEMORY\n");
        exit(0);
    }
}


void check_1D_mem(double *allocation)
{
    if (allocation == NULL)
    {
        printf("NO MEMORY\n");
        exit(0);
    }
}


void link_ptr_to_values(double **arr, double *buf, int len_of_stripe, int len)
{
    int i;
    for (i = 0; i < len; i++)
    {
        arr[i] = buf + len_of_stripe*i;
        printf("aaaaa: %d ", (len_of_stripe*i));
    }
    printf("\n");
}

/*
NOTE: child, father and rank are used as synonyms for process
*/
int main(int argc, char** argv)
{
    int total_dmnsn=argc;
    char *str = argv[1];
    total_dmnsn=atoi(str);
    int dimension = total_dmnsn;//n of rows and columns
    double precision = 0.01f;
    int world_size;
    int world_rank;
    int rc = MPI_Init(NULL, NULL);
    if (rc != MPI_SUCCESS) {
        printf ("Error starting MPI program\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
    }
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    if(DEBUG==1)
    {
        printf("size: %d\n",world_size);
        printf("rank: %d\n",world_rank);
        printf("\n");
    }
    // father process
    if (world_rank == 0)
    {
        struct timeval strt, nd;
        gettimeofday(&strt, 0);
        double *square_array = initialise_square_array(dimension);
        int div= (int)ceil(((float)(dimension-2))/(world_size-1));
        int n_rows_per_chunk = div+2;
        int r = (world_size-1), c = (dimension*n_rows_per_chunk), i, j, count; 
        //chunk is the array on which a process works on
        double *chunk[r];
        for (i=0; i<r; i++)
            chunk[i] = (double *)malloc((unsigned long)c * sizeof(double));
        count = 0; 
        for (i = 0; i <  r; i++) 
            for (j = 0; j < c; j++) 
                chunk[i][j] = i*c+j;
        //guard for edge case
        if(get_range_from_p_num_start(dimension-2, world_size-1, 1)==(-1))
        {
            exit(0);
        }
        int start=0;
        int end=0;
        //SENDING loop: father knows exactly what to send to each child, so he
        //does it in the below loop
        for (i = 0; i <  r; i++) 
        {
            start=get_range_from_p_num_start(dimension-2, world_size-1, i+1);
            end=get_range_from_p_num_end(dimension-2, world_size-1, i+1);
            start=start-1;//get edge top
            end=end+1;//get edge bottom
            chunk[i] = select_chunk(dimension, square_array, start,
                                    n_rows_per_chunk);
            //the height of the chunk is multiplied by the dimension of the
            //array since we are representing the 2D array through a 1D one
            int entire_dimension=dimension*(end-start+1);
            //send dimension
            MPI_Send(&entire_dimension, 1, MPI_INT, i+1, 321, MPI_COMM_WORLD);
            //send actual data, now that the child knows the correct length
            MPI_Send(chunk[i], entire_dimension, MPI_DOUBLE, i+1, 321,
                     MPI_COMM_WORLD);
        }
        //RECEIVING loop: father gets stuck in this loop until he starts
        //hearing from his children
        for (i = 0; i <  r; i++) 
        {
            //father needs to re-compute start and end of each child to know
            //where to put all their responses
            start=get_range_from_p_num_start(dimension-2, world_size-1, i+1);
            end=get_range_from_p_num_end(dimension-2, world_size-1, i+1);
            start=start-1;//get edge top
            end=end+1;//get edge bottom
            if(DEBUG==1)
            {
                printf("Reading process %d soon\n",i);
            }
            //gets stuck here
            MPI_Recv(chunk[i], dimension*(end-start+1), MPI_DOUBLE, i+1, 123,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if(DEBUG==1)
            {
                printf("Done %d\n",i);
            }
            //updates main array with computed chunks here
            square_array = stitch_array(square_array, chunk[i],
                                        start, end, dimension);
        }
        gettimeofday(&nd, 0);
        long a=((nd.tv_sec * 1000000 + nd.tv_usec) - 
                (strt.tv_sec * 1000000 + strt.tv_usec))/1000;
        printf("\ntime: %ld\n", a);
        printf("dim: %d\nn_processes: %d\n",dimension,world_size);
        //print_array(dimension, square_array);
        MPI_Finalize();
        exit(0);
    }
    else
        // child process
    {
        int i;
        for (i = 1; i < world_size; i++)
        {
            //condition to identify the single, correct, rank
            if (world_rank == i)
            {
                int my_dimension;//number of items in the chunk
                int flag=0;//flag for checking "Isend" & "Irecv" buffers
                int flag_king=0;//flag for checking "Isend" & "Irecv" buffers
                //called "king" because belonging to processes "sitting" on top
                //of others
                MPI_Request reqs[2];//two request variables for the two pairs
                //of "Isend" & "Irecv"
                MPI_Status statuus;
                MPI_Status statuuus;//two statuses again, to differentiate
                //receive the assigned dimension from father
                MPI_Recv(&my_dimension, 1, MPI_INT, 0, 321, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
                double *my_chunk1 = malloc((unsigned long)
                                           my_dimension * sizeof(double));
                //receive the actual chunk from father
                MPI_Recv(my_chunk1, my_dimension, MPI_DOUBLE, 0, 321,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                int start=get_range_from_p_num_start(dimension-2,
                                                     world_size-1, i);
                int end=get_range_from_p_num_end(dimension-2, world_size-1, i);
                start=start-1;//get edge top
                end=end+1;//get edge bottom
                int n_rows_per_chunk=(end-start+1);
                //just debugging
                if(DEBUG==1)
                {
                    printf("Hello world from processor #%d out of %d "
                           "processors\n", world_rank, world_size);
                    printf("A #%d \n", (dimension-2));
                    printf("B #%d \n", (world_size-1));
                    printf("C #%d \n", (i));
                    printf("start #%d \n", start);
                    printf("end #%d \n", end);
                    printf("dimension #%d \n", dimension);
                    printf("n_rows_per_chunk #%d \n", n_rows_per_chunk);
                }
                int k,j,prec_counter=0;
                double prev_val=0;
                while (prec_counter<my_dimension)
                {
                    //means: don't do any of below if there is just 1 child
                    if(world_size>=3)
                    {
                        //if the current process is NOT the last process
                        //then send the LAST row to the NEXT process
                        if(i!=(world_size-1))
                        {
                            //SEND BELOW
                            double *row_to_pass = malloc((unsigned long)
                                                dimension * sizeof(double));
                            int m;
                            for (m = 0; m < dimension; m++)
                            {
                                row_to_pass[m]=my_chunk1[(n_rows_per_chunk-2)*
                                                         (dimension)+m];
                            }
                            MPI_Isend(row_to_pass, dimension, MPI_DOUBLE,
                                      (i+1), tag2, MPI_COMM_WORLD, &reqs[1]);
                        }
                        else //if the current process IS the LAST process
                            //then send the FIRST row to the PREVIOUS process
                        {
                            //SEND ABOVE
                            double *row_to_pass = malloc((unsigned long)
                                                dimension * sizeof(double));
                            int m;
                            for (m = 0; m < dimension; m++)
                            {
                                row_to_pass[m]=my_chunk1[(0+1)*(dimension)+m];
                            }
                            MPI_Isend(row_to_pass, dimension, MPI_DOUBLE,
                                      (i-1), tag1, MPI_COMM_WORLD, &reqs[0]);
                        }
                        //if the current process is NOT the first process
                        //then send the FIRST row to the PREVIOUS process
                        if(i!=1)
                        {
                            //SEND ABOVE
                            double *row_to_pass = malloc((unsigned long)
                                                dimension * sizeof(double));
                            int m;
                            for (m = 0; m < dimension; m++)
                            {
                                row_to_pass[m]=my_chunk1[(0+1)*(dimension)+m];
                            }
                            MPI_Isend(row_to_pass, dimension, MPI_DOUBLE,
                                      (i-1), tag1, MPI_COMM_WORLD, &reqs[0]);
                        }
                        else //if the current process IS the FIRST process
                            //then send the LAST row to the NEXT process
                        {
                            //SEND BELOW
                            double *row_to_pass = malloc((unsigned long)
                                                dimension * sizeof(double));
                            int m;
                            for (m = 0; m < dimension; m++)
                            {
                                row_to_pass[m]=my_chunk1[(n_rows_per_chunk-2)*
                                                         (dimension)+m];
                            }
                            MPI_Isend(row_to_pass, dimension, MPI_DOUBLE,
                                      (i+1), tag2, MPI_COMM_WORLD, &reqs[1]);
                        }
                    }
                    ///////////////////////////////////////////////////////////
                    ///////////       real computation           //////////////
                    ///////////////////////////////////////////////////////////
                    for (k = 1; k < n_rows_per_chunk-1; k++)
                    {
                        for (j = 1; j < dimension-1; j++)
                        {
                            prev_val=my_chunk1[k*(dimension)+j];
                            my_chunk1[k*(dimension)+j] = (my_chunk1[(k+1)*
                                (dimension)+j]+my_chunk1[(k-1)*(dimension)+j]+
                                my_chunk1[k*(dimension)+(j+1)]+my_chunk1[k*
                                                                                                   (dimension)+(j-1)])/4;
                            if(fabs(prev_val-my_chunk1[k*(dimension)+j])<
                               precision)
                            {
                                prec_counter++;
                            }
                            else
                            {
                                prec_counter=0;
                            }
                        }
                    }
                    ///////////////////////////////////////////////////////////
                    ///////////       real computation           //////////////
                    ///////////////////////////////////////////////////////////
                    if(world_size>=3)
                    {
                        //if the current process is NOT the last process
                        //listen the NEXT to fill the LAST row of this
                        if(i!=(world_size-1))
                        {
                            //FROM BELOW
                            double *row_to_receive = malloc((unsigned long)
                                                dimension * sizeof(double));
                            int m;
                            for (m = 0; m < dimension; m++) 
                            {
                                row_to_receive[m]=0;
                            }
                            MPI_Irecv(row_to_receive, dimension, MPI_DOUBLE,
                                      (i+1), tag1, MPI_COMM_WORLD, &reqs[0]);
                            /* the below 12 lines are fundamental for this
                            program, so let us explain:
                            "MPI_Test" tests the buffer of the "channel" to see
                            if something is there. If something is there, then
                            update the row, if nothing is there, proceed
                            without updating
                            */
                            MPI_Test(&reqs[0], &flag, &statuus);
                            if(flag==1)
                            {
                                int n;
                                for (n = 0; n < dimension; n++)
                                {
                                    my_chunk1[(n_rows_per_chunk-1)*
                                              (dimension)+n]=row_to_receive[n];
                                }
                                flag=0;
                            }
                            else{}
                        }
                        else //if the current process IS the last process
                            //listen the PREVIOUS to fill the FIRST row of this
                        {
                            //FROM ABOVE
                            double *row_to_receive = malloc((unsigned long)
                                                dimension * sizeof(double));
                            int m;
                            for (m = 0; m < dimension; m++) 
                            {
                                row_to_receive[m]=0;
                            }
                            MPI_Irecv(row_to_receive, dimension, MPI_DOUBLE,
                                      (i-1), tag2, MPI_COMM_WORLD, &reqs[1]);
                            /*
                            "MPI_Test" tests the buffer of the "channel" to see
                            if something is there. If something is there, then
                            update the row, if nothing is there, proceed
                            without updating
                            */
                            MPI_Test(&reqs[1], &flag_king, &statuuus);
                            if(flag_king==1)
                            {
                                int n;
                                for (n = 0; n < dimension; n++)
                                {
                                    my_chunk1[(0)*(dimension)+n]=
                                        row_to_receive[n];
                                }
                                flag_king=0;
                            }
                            else{}
                        }
                        //if the current process is NOT the first process
                        //listen the PREVIOUS to fill the FIRST row of this
                        if(i!=1)
                        {
                            //FROM ABOVE
                            double *row_to_receive = malloc((unsigned long)
                                                dimension * sizeof(double));
                            int m;
                            for (m = 0; m < dimension; m++) 
                            {
                                row_to_receive[m]=0;
                            }
                            MPI_Irecv(row_to_receive, dimension, MPI_DOUBLE,
                                      (i-1), tag2, MPI_COMM_WORLD, &reqs[1]);
                            /*
                            "MPI_Test" tests the buffer of the "channel" to see
                            if something is there. If something is there, then
                            update the row, if nothing is there, proceed
                            without updating
                            */
                            MPI_Test(&reqs[1], &flag_king, &statuuus);
                            if(flag_king==1)
                            {
                                int n;
                                for (n = 0; n < dimension; n++)
                                {
                                    my_chunk1[(0)*(dimension)+n]=
                                        row_to_receive[n];
                                }
                                flag_king=0;
                            }
                            else{}
                        }
                        else //if the current process IS the first process
                            //listen the NEXT to fill the LAST row of this
                        {
                            //FROM BELOW
                            double *row_to_receive = malloc((unsigned long)
                                                dimension * sizeof(double));
                            int m;
                            for (m = 0; m < dimension; m++) 
                            {
                                row_to_receive[m]=0;
                            }
                            MPI_Irecv(row_to_receive, dimension, MPI_DOUBLE,
                                      (i+1), tag1, MPI_COMM_WORLD, &reqs[0]);
                            /*
                            "MPI_Test" tests the buffer of the "channel" to see
                            if something is there. If something is there, then
                            update the row, if nothing is there, proceed
                            without updating
                            */
                            MPI_Test(&reqs[0], &flag, &statuus);
                            if(flag==1)
                            {
                                int n;
                                for (n = 0; n < dimension; n++)
                                {
                                    my_chunk1[(n_rows_per_chunk-1)*
                                              (dimension)+n]=row_to_receive[n];
                                }
                                flag=0;
                            }
                            else{}
                        }
                    }
                }
                //success message the child process outputs when he is done
                printf("\nProcess number %d SUCCESSFULLY completed tasks and"
                       " exited\n", i);
                if(DEBUG==1)
                {
                    for (k = 1; k < n_rows_per_chunk-1; k++) {
                        for (j = 1; j < dimension-1; j++) {
                            printf("%d", j);
                        }
                        printf("\n");
                    }
                    print_non_square_array((end-start+1),dimension,my_chunk1,
                                           i);
                    printf("\n\n\n");
                }
                //send the completed chunk to the father
                MPI_Send(my_chunk1, my_dimension, MPI_DOUBLE, 0, 123,
                         MPI_COMM_WORLD);
            }
        }
    }
    MPI_Finalize();
}